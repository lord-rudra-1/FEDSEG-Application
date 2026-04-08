# FEDSEG-APP

Federated medical image segmentation demo built with a `FastAPI` backend and a `Streamlit` frontend.  
The app performs chest X-ray region-of-interest segmentation using a ResNet-UNet style model and provides visual overlays, confidence analytics, and exportable outputs.

## Features

- Upload an image and run segmentation inference through a REST API
- Interactive UI with original image, predicted mask, and overlay view
- Confidence analytics (histograms, confidence bands, 3D volume rendering)
- Export artifacts: mask PNG, overlay PNG, and attributes CSV
- Health-check endpoint for backend availability
- Safer API contract with request validation and structured responses
- Compressed mask transport (`PNG` encoded as base64) to reduce payload size
- Modularized frontend utilities for rendering, image processing, and API access
- Basic automated tests for backend and frontend helper logic

## Project Structure

```text
FEDSEG-APP/
  backend/
    api.py                 # FastAPI application and inference endpoint
    config.py              # Environment-driven runtime settings
    schemas.py             # Pydantic response models
  frontend/
    app.py                 # Streamlit user interface
    api_client.py          # HTTP client and payload decoding
    image_utils.py         # Image processing helpers
    styles.py              # CSS theme injection
    ui.py                  # UI rendering helpers
  model/
    model.py               # ResNet-UNet model definition
    infer.py               # Model loading and prediction pipeline
    fedseg_model.pt        # Trained model checkpoint (required, not included by default)
  tests/
    test_api.py            # API and payload contract tests
    test_frontend_utils.py # Frontend helper tests
  utils/
    __init__.py
  requirements.txt
```

## Tech Stack

- `Python`
- `FastAPI` + `Uvicorn`
- `Streamlit`
- `PyTorch` + `torchvision`
- `OpenCV`
- `Plotly`
- `NumPy`, `Pandas`, `requests`
- `pytest`, `httpx`, `pydantic`

## Prerequisites

- Python `3.9+` recommended
- `pip`
- Model checkpoint at `model/fedseg_model.pt`

## Installation

1. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

## Model Checkpoint Setup

This project expects a trained model checkpoint at:

- `model/fedseg_model.pt`

If you trained the model in your notebook, save it using:

```python
torch.save(global_model.state_dict(), "model/fedseg_model.pt")
```

> If the file is missing, the backend will fail with a `FileNotFoundError`.

## Running the Application

Open two terminals from the project root.

### 1) Start backend API

```powershell
python -m uvicorn backend.api:app --reload
```

Backend default URL: `http://localhost:8000`  
Health endpoint: `GET /health`

### 2) Start Streamlit frontend

```powershell
streamlit run frontend/app.py
```

Frontend default URL is shown in the terminal (usually `http://localhost:8501`).

## API Reference

### `GET /health`

Returns backend availability status.

**Response**

```json
{ "status": "ok" }
```

### `POST /predict`

Runs segmentation on an uploaded image.

**Request**

- `multipart/form-data`
- Field: `file` (`UploadFile`)

**Response (shape)**

```json
{
  "image_height": 1024,
  "image_width": 1024,
  "inference_time_ms": 42.13,
  "mask_mean": 0.137,
  "mask_std": 0.223,
  "mask_min": 0.0,
  "mask_max": 0.998,
  "mask_png_base64": "<base64-encoded-png>",
  "mask_encoding": "png_base64"
}
```

## Configuration

Frontend supports overriding backend URL through:

- Environment variable: `FEDSEG_API_URL`
- Sidebar input: API URL field

Backend supports additional environment variables:

- `FEDSEG_MODEL_PATH`
- `FEDSEG_INFERENCE_IMAGE_SIZE`
- `FEDSEG_MAX_UPLOAD_SIZE_BYTES`
- `FEDSEG_REQUEST_TIMEOUT_SECONDS`

Default backend URL is `http://localhost:8000`.

## Troubleshooting

- **Backend not reachable in UI**  
  Ensure `uvicorn` is running and API URL matches backend host/port.

- **Model not found error**  
  Confirm `model/fedseg_model.pt` exists and is readable.

- **Image decode error**  
  Upload a valid PNG/JPEG image file.

- **Large uploads rejected**  
  Increase `FEDSEG_MAX_UPLOAD_SIZE_BYTES` if you intentionally need larger input files.

- **Slow inference**  
  Verify GPU availability and correct PyTorch/CUDA installation.

## Limitations

- Demo/research implementation only
- Not validated for clinical workflows
- No authentication/rate-limiting layer in API

## Testing

Run the lightweight regression suite from the project root:

```powershell
pytest
```

## Disclaimer

This software is for educational and research demonstration purposes only.  
It is **not** a medical device and must **not** be used for clinical diagnosis or treatment decisions.

