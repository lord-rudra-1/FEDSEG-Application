import base64
import logging
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.config import settings
from backend.schemas import HealthResponse, PredictResponse
from model.infer import predict

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        app_version=settings.app_version,
    )


def _validate_upload(file: UploadFile, contents: bytes) -> None:
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds {settings.max_upload_size_bytes} bytes.",
        )
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use a PNG or JPEG image.",
        )


def _encode_mask_png(mask: np.ndarray) -> str:
    mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    ok, buffer = cv2.imencode(".png", mask_uint8)
    if not ok:
        raise RuntimeError("Failed to encode output mask.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


@app.post("/predict", response_model=PredictResponse)
async def predict_api(file: UploadFile = File(...)) -> PredictResponse:
    contents = await file.read()
    _validate_upload(file, contents)
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Use a valid PNG or JPEG.")

    h, w = img.shape[:2]
    t0 = time.perf_counter()
    try:
        mask = predict(img)
    except FileNotFoundError as exc:
        logger.exception("Model checkpoint was not found.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected inference failure.")
        raise HTTPException(status_code=500, detail="Inference failed unexpectedly.") from exc

    inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    mask_flat = np.array(mask).flatten()
    logger.info(
        "Processed image %s (%sx%s) in %.2f ms",
        file.filename or "<memory>",
        w,
        h,
        inference_time_ms,
    )
    return PredictResponse(
        image_height=h,
        image_width=w,
        inference_time_ms=inference_time_ms,
        mask_mean=float(np.mean(mask_flat)),
        mask_std=float(np.std(mask_flat)),
        mask_min=float(np.min(mask_flat)),
        mask_max=float(np.max(mask_flat)),
        mask_png_base64=_encode_mask_png(mask),
    )
