from fastapi import APIRouter, HTTPException, File, UploadFile
import torch
import torchaudio
import os
import logging
import ffmpeg
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ForSequenceClassification

router = APIRouter()

# Logging
logging.basicConfig(level=logging.DEBUG)

# Log torchaudio version and available backends
logging.debug(f"torchaudio version: {torchaudio.__version__}")
logging.debug(f"Available backends: {torchaudio.list_audio_backends()}")

# Try to set the audio backend to ffmpeg, fall back to soundfile
try:
    torchaudio.set_audio_backend("ffmpeg")
except RuntimeError:
    logging.warning("FFmpeg backend not available, falling back to soundfile")
    torchaudio.set_audio_backend("soundfile")

# Config
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR  = 16_000
HOP_LENGTH = 320

# Load CTC Model for Forced Alignment
CTC_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
ctc_proc     = Wav2Vec2Processor.from_pretrained(CTC_MODEL_ID)
ctc_model    = Wav2Vec2ForCTC.from_pretrained(CTC_MODEL_ID).to(DEVICE)

# Load Classifier Model
CLF_MODEL_DIR = "/Users/mrmacbook/projects/nibras_api/model"
clf_proc      = Wav2Vec2Processor.from_pretrained(CLF_MODEL_DIR)
clf_model     = Wav2Vec2ForSequenceClassification.from_pretrained(CLF_MODEL_DIR).to(DEVICE)

# Log the model configuration
logging.debug(f"Wav2Vec2 model config: {clf_model.config}")

# Label Mapping
ID2LABEL = {
    0: "W",
    1: "S",
    2: "PH",
    3: "PR",
    4: "none"
}

# Audio Loader
def load_waveform(path, start_s=None, end_s=None):
    wav, sr = torchaudio.load(path)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
    if start_s is not None and end_s is not None:
        s, e = int(start_s * TARGET_SR), int(end_s * TARGET_SR)
        wav = wav[:, s:e]
    
    # Ensure the waveform is at least 0.25 seconds long (4000 samples at 16 kHz)
    min_length = 4000  # 0.25 seconds at 16 kHz
    if wav.size(1) < min_length:
        padding_length = min_length - wav.size(1)
        wav = torch.nn.functional.pad(wav, (0, padding_length))
        logging.debug(f"Padded waveform from {wav.size(1) - padding_length} to {min_length} samples")
    
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0).numpy()

# Convert file to WAV format
def convert_to_wav(input_path: str, output_path: str):
    try:
        ffmpeg.input(input_path).output(output_path, format='wav', acodec='pcm_s16le', ar=16000).run(overwrite_output=True)
        logging.debug(f"Converted {input_path} to {output_path}")
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        raise

# Get Word Segments
def get_word_segments(audio_path):
    sig = load_waveform(audio_path)
    inputs = ctc_proc(sig, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = ctc_model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    
    decoded = ctc_proc.decode(pred_ids[0], skip_special_tokens=True)
    words = decoded.split()
    
    segments = []
    total_duration = len(sig) / TARGET_SR  # Total audio duration in seconds
    if words:
        # Ensure each segment is at least 0.25 seconds long
        min_segment_duration = 0.25  # Minimum duration in seconds
        segment_duration = max(total_duration / len(words), min_segment_duration)
        for i, word in enumerate(words):
            start = i * segment_duration
            end = min((i + 1) * segment_duration, total_duration)
            # Ensure the segment is at least min_segment_duration long
            if end - start < min_segment_duration:
                end = start + min_segment_duration
                if end > total_duration:
                    start = max(0, total_duration - min_segment_duration)
                    end = total_duration
            segments.append({"word": word, "start": start, "end": end})
    
    return segments

# Classify Word Slice
def classify_word(audio_path, start_frame, end_frame):
    start_s = start_frame * HOP_LENGTH / TARGET_SR
    end_s   = end_frame   * HOP_LENGTH / TARGET_SR
    sig     = load_waveform(audio_path, start_s, end_s)
    
    logging.debug(f"Raw waveform length: {len(sig)} samples")
    
    inputs = clf_proc(sig, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    input_values = inputs["input_values"]
    logging.debug(f"Processed input shape before padding: {input_values.shape}")

    # Ensure the processed input is long enough for the model
    min_required_length = 2048  # Increased to account for downsampling
    if input_values.shape[-1] < min_required_length:
        padding_length = min_required_length - input_values.shape[-1]
        input_values = torch.nn.functional.pad(input_values, (0, padding_length))
        logging.debug(f"Padded processed input from {input_values.shape[-1] - padding_length} to {min_required_length} samples")
    
    logging.debug(f"Final input shape: {input_values.shape}")
    inputs["input_values"] = input_values
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = clf_model(**inputs).logits
    pred_id = torch.argmax(logits, dim=-1).item()
    return ID2LABEL[pred_id]

# Assess Word-Level
def assess_word_level(audio_path):
    segments       = get_word_segments(audio_path)
    total_words    = len(segments)
    stutter_events = 0
    durations      = []
    labeled        = []

    for seg in segments:
        label = classify_word(audio_path, seg["start"], seg["end"])
        start_s = seg["start"] * HOP_LENGTH / TARGET_SR
        end_s   = seg["end"]   * HOP_LENGTH / TARGET_SR
        labeled.append({
            "word": seg["word"],
            "start_s": round(start_s, 3),
            "end_s":   round(end_s,   3),
            "label":   label
        })
        if label != "none":
            stutter_events += 1
            durations.append(end_s - start_s)

    frequency_pct = (stutter_events / total_words * 100) if total_words else 0.0
    total_dur     = round(sum(durations), 3)
    avg_dur       = round((total_dur / stutter_events) if stutter_events else 0.0, 3)

    summary = {
        "total_words":       total_words,
        "stutter_events":    stutter_events,
        "frequency_%":       round(frequency_pct, 2),
        "total_duration_s":  total_dur,
        "average_duration_s": avg_dur
    }
    return summary, labeled

# API Endpoint: /assess
@router.post("/assess/")
async def assess(file: UploadFile = File(...)):
    try:
        temp_file_path = f"/tmp/{file.filename}"
        converted_file_path = f"/tmp/converted_{file.filename}.wav"

        # Save the uploaded file
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            logging.debug(f"File saved at {temp_file_path}, size: {len(content)} bytes")

        # Convert the file to WAV format
        convert_to_wav(temp_file_path, converted_file_path)

        summary, segments = assess_word_level(converted_file_path)

        os.remove(temp_file_path)
        os.remove(converted_file_path)
        return {
            "summary": summary,
            "segments": segments
        }

    except Exception as e:
        logging.exception("Assessment error")
        raise HTTPException(status_code=500, detail=str(e))
