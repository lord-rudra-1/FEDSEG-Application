from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    app_name: str
    app_version: str


class PredictResponse(BaseModel):
    image_height: int
    image_width: int
    inference_time_ms: float
    mask_mean: float
    mask_std: float
    mask_min: float
    mask_max: float
    mask_png_base64: str = Field(..., description="PNG-encoded grayscale mask as a base64 string.")
    mask_encoding: str = Field(default="png_base64")
