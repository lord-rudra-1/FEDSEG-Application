from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import time
from model.infer import predict

app = FastAPI(title="FEDSEG API", version="1.0")

@app.get("/health")
def health():
    """For frontend to check if backend is reachable."""
    return {"status": "ok"}

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Use a valid PNG or JPEG.")

    h, w = img.shape[:2]
    t0 = time.perf_counter()
    mask = predict(img)
    inference_time_ms = (time.perf_counter() - t0) * 1000

    mask_flat = np.array(mask).flatten()
    return {
        "mask": mask.tolist(),
        "image_height": h,
        "image_width": w,
        "inference_time_ms": round(inference_time_ms, 2),
        "mask_mean": float(np.mean(mask_flat)),
        "mask_std": float(np.std(mask_flat)),
        "mask_min": float(np.min(mask_flat)),
        "mask_max": float(np.max(mask_flat)),
    }
