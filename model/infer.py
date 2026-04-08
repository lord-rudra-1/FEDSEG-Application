"""
Inference for FEDSEG: ResNet-UNet from final-fedrated-pro.ipynb.
Uses model/fedseg_model.pt — save from notebook: torch.save(global_model.state_dict(), "model/fedseg_model.pt").
Falls back to segmentation_models_pytorch Unet if state_dict keys don't match.
"""
import logging
import os

import cv2
import numpy as np
import torch

from model.config import INFERENCE_IMAGE_SIZE, MODEL_PATH
from model.model import ResNetUNet

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "fedseg_model.pt")


def load_model():
    model_path = MODEL_PATH or DEFAULT_MODEL_PATH
    path = os.path.abspath(model_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model not found: {path}. "
            "From the notebook run: torch.save(global_model.state_dict(), 'model/fedseg_model.pt')"
        )
    state = torch.load(path, map_location=device)

    try:
        model = ResNetUNet(pretrained_encoder=False)
        model.load_state_dict(state, strict=True)
        logger.info("Loaded custom ResNetUNet checkpoint from %s", path)
    except Exception as exc:
        logger.warning("Custom checkpoint load failed, trying SMP Unet fallback: %s", exc)
        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )
        model.load_state_dict(state, strict=True)
        logger.info("Loaded SMP Unet fallback checkpoint from %s", path)

    model.to(device)
    model.eval()
    return model


# Load at import; optional lazy load if you prefer
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def predict(image):
    """
    image: BGR numpy (H, W, 3) from cv2.imdecode.
    Returns: float32 mask (H, W) in [0, 1], resized to input size.
    """
    model = get_model()
    h, w = image.shape[:2]

    # Match notebook QaTaDataset: RGB, 224x224, [0,1], CHW
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INFERENCE_IMAGE_SIZE, INFERENCE_IMAGE_SIZE))
    img = (img / 255.0).astype(np.float32)
    img = img.transpose(2, 0, 1)
    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        # ResNetUNet returns sigmoid; smp Unet(activation=None) returns logits
        if not hasattr(model, "encoder0"):
            out = torch.sigmoid(out)

    mask = out.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return mask.astype(np.float32)
