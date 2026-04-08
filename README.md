# FEDSEG-APP

Federated medical image segmentation demo built with a `FastAPI` backend and a `Streamlit` frontend.  
The app performs chest X-ray region-of-interest segmentation using a ResNet-UNet style model and provides visual overlays, confidence analytics, and exportable outputs.

## Features

- Upload an image and run segmentation inference through a REST API
- Interactive UI with original image, predicted mask, and overlay view
- Confidence analytics (histograms, confidence bands, 3D volume rendering)
- Export artifacts: mask PNG, overlay PNG, and attributes CSV
- Health-check endpoint for backend availability

## Project Structure

```text
FEDSEG-APP/
  backend/
    api.py                 # FastAPI application and inference endpoint
  frontend/
    app.py                 # Streamlit user interface
  model/
    model.py               # ResNet-UNet model definition
    infer.py               # Model loading and prediction pipeline
    fedseg_model.pt        # Trained model checkpoint (required, not included by default)
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
  "mask": [[0.0, 0.1, 0.8], [0.2, 0.6, 0.9]],
  "image_height": 1024,
  "image_width": 1024,
  "inference_time_ms": 42.13,
  "mask_mean": 0.137,
  "mask_std": 0.223,
  "mask_min": 0.0,
  "mask_max": 0.998
}
```

## Configuration

Frontend supports overriding backend URL through:

- Environment variable: `FEDSEG_API_URL`
- Sidebar input: API URL field

Default backend URL is `http://localhost:8000`.

## Troubleshooting

- **Backend not reachable in UI**  
  Ensure `uvicorn` is running and API URL matches backend host/port.

- **Model not found error**  
  Confirm `model/fedseg_model.pt` exists and is readable.

- **Image decode error**  
  Upload a valid PNG/JPEG image file.

- **Slow inference**  
  Verify GPU availability and correct PyTorch/CUDA installation.

## Limitations

- Demo/research implementation only
- Not validated for clinical workflows
- No authentication/rate-limiting layer in API

## Disclaimer

This software is for educational and research demonstration purposes only.  
It is **not** a medical device and must **not** be used for clinical diagnosis or treatment decisions.

