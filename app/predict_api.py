from fastapi import APIRouter, HTTPException, File, UploadFile
import torch
import torchaudio
import os
import logging
import ffmpeg
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

router = APIRouter()

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

# Model Path
model_path = "/Users/mrmacbook/projects/nibras_api/model"

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define label mapping
ID2LABEL = {
    0: "W",
    1: "S",
    2: "PH",
    3: "PR",
    4: "none"
}

# Audio Preprocessing
def preprocess_audio(audio_path: str):
    waveform, sr = torchaudio.load(audio_path)
    target_sampling_rate = processor.feature_extractor.sampling_rate

    if sr != target_sampling_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sampling_rate)(waveform)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    
    waveform = waveform.unsqueeze(0).squeeze()
    logging.debug(f"Shape of preprocessed audio: {waveform.shape}")

    return waveform

# Convert file to WAV format
def convert_to_wav(input_path: str, output_path: str):
    try:
        ffmpeg.input(input_path).output(output_path, format='wav', acodec='pcm_s16le', ar=16000).run(overwrite_output=True)
        logging.debug(f"Converted {input_path} to {output_path}")
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        raise

# Predict Endpoint
@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
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

        # Preprocess the converted file
        waveform = preprocess_audio(converted_file_path)
        inputs = processor(waveform, return_tensors="pt", sampling_rate=processor.feature_extractor.sampling_rate)
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_label = ID2LABEL[predicted_id]

        # Clean up temporary files
        os.remove(temp_file_path)
        os.remove(converted_file_path)

        return {"prediction": predicted_label}

    except Exception as e:
        logging.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))
