import os
from dataclasses import dataclass


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("FEDSEG_APP_NAME", "FEDSEG API")
    app_version: str = os.getenv("FEDSEG_APP_VERSION", "2.0")
    model_path: str = os.getenv("FEDSEG_MODEL_PATH", "")
    inference_image_size: int = _read_int("FEDSEG_INFERENCE_IMAGE_SIZE", 224)
    max_upload_size_bytes: int = _read_int("FEDSEG_MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024)
    request_timeout_seconds: float = _read_float("FEDSEG_REQUEST_TIMEOUT_SECONDS", 60.0)


settings = Settings()
