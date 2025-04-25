from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging, io
import numpy as np
from pydub import AudioSegment
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

router = APIRouter()
logging.basicConfig(level=logging.INFO)

# ⚡ Load model from local model/ folder
MODEL_PATH = "model"

try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH, local_files_only=True)
    model     = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
except Exception as e:
    raise RuntimeError(f"Model load failed: {e}")

# ✅ Ensure ID2LABEL uses integer keys
ID2LABEL = {int(k): v for k, v in model.config.id2label.items()}

@router.post("/", summary="Predict stutter type from uploaded audio")
async def predict_stutter_type(audio_file: UploadFile = File(...)):
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    try:
        data = await audio_file.read()
        ext  = audio_file.filename.rsplit(".", 1)[-1].lower()
        seg  = AudioSegment.from_file(io.BytesIO(data), format=ext)
        seg  = seg.set_frame_rate(16000).set_channels(1)
        samples = np.array(seg.get_array_of_samples())
        audio_np = samples.astype(np.float32) / 32768.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio loading failed: {e}")

    try:
        inputs = processor(audio_np, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        pred_idx = int(torch.argmax(logits, dim=-1).cpu().item())
        pred_label = ID2LABEL.get(pred_idx, "Unknown")
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed")

    return JSONResponse({
        "prediction": pred_label,
        "prediction_index": pred_idx,
        "confidence": round(probs[pred_idx], 4),
        "all_confidences": {ID2LABEL[i]: round(probs[i], 4) for i in range(len(probs))}
    })
