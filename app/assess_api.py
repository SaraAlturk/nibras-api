from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io, logging
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

router = APIRouter()
logging.basicConfig(level=logging.INFO)

# ⚡ Load model from local /model folder
MODEL_PATH = "./model"

try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model     = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Correct label map
label_map = {int(k): v for k, v in model.config.id2label.items()}

@router.post("/", summary="Assess stuttering in uploaded audio")
async def assess_stuttering(file: UploadFile = File(...)):
    if not file or "." not in file.filename:
        raise HTTPException(400, "Invalid or missing file.")
    ext = file.filename.rsplit(".", 1)[1].lower()

    try:
        data = await file.read()
        seg = AudioSegment.from_file(io.BytesIO(data), format=ext)
    except Exception as e:
        raise HTTPException(400, f"Bad audio: {e}")
    
    seg = seg.set_frame_rate(16000).set_channels(1)
    samples = np.array(seg.get_array_of_samples())
    sw = seg.sample_width
    if sw == 2:
        audio_np = samples.astype(np.float32) / 32768.0
    elif sw == 4:
        audio_np = samples.astype(np.float32) / (2**31)
    else:
        audio_np = samples.astype(np.float32)
        if sw == 1:
            audio_np = (audio_np - 128) / 128.0
    if audio_np.size == 0:
        raise HTTPException(400, "Empty audio.")

    # Sliding window inference
    WIN, HOP, SR = 0.2, 0.1, 16000
    wlen = int(WIN * SR)
    hop  = int(HOP * SR)
    if len(audio_np) < wlen:
        audio_np = np.pad(audio_np, (0, wlen - len(audio_np)))

    pred_labels = []
    for start in range(0, len(audio_np) - wlen + 1, hop):
        chunk = audio_np[start : start + wlen]
        inputs = processor(chunk, sampling_rate=SR, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        pid = torch.argmax(logits, dim=-1)[0].item()
        pred_labels.append(label_map.get(pid, "none"))

    # Count events
    num_events = 0
    stutter_frames = 0
    prev = "none"
    for lbl in pred_labels:
        if lbl != "none":
            stutter_frames += 1
            if prev == "none" or prev != lbl:
                num_events += 1
        prev = lbl

    frame_duration = WIN
    stuttering_time = stutter_frames * frame_duration
    total_duration = len(audio_np) / SR

    # Word segmentation
    nonsilent_ms = detect_nonsilent(
        seg,
        min_silence_len=100,
        silence_thresh=seg.dBFS - 10
    )
    word_intervals = [(start/1000.0, end/1000.0) for start, end in nonsilent_ms]
    total_words = max(len(word_intervals), 1)

    stuttered_word_idxs = set()
    for i, lbl in enumerate(pred_labels):
        if lbl == "none":
            continue
        mid_time = (i + 0.5) * frame_duration
        for w_idx, (w_start, w_end) in enumerate(word_intervals):
            if w_start <= mid_time <= w_end:
                stuttered_word_idxs.add(w_idx)
                break

    num_stuttered_words = len(stuttered_word_idxs)
    word_level_freq = (num_stuttered_words / total_words) * 100.0

    event_durations = [(f1 - f0 + 1) * frame_duration
                       for f0, f1 in [(i, i) for i, l in enumerate(pred_labels) if l != "none"]]
    top3 = sorted(event_durations, reverse=True)[:3]
    avg_top3 = sum(top3) / len(top3) if top3 else 0.0
    ssi4_score = word_level_freq + (avg_top3 * 100.0)

    result = {
        "total_words": total_words,
        "stuttered_words": num_stuttered_words,
        "word_level_frequency_percent": round(word_level_freq, 2),
        "stuttering_duration_seconds": round(stuttering_time, 2),
        "num_events": num_events,
        "ssi4_score": round(ssi4_score, 2),
    }
    logging.info(f"Returning result: {result}")
    return JSONResponse(result)
