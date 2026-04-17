"""FastAPI application for plant disease classification.

Run locally:
    uvicorn api.main:app --reload

Swagger docs at /docs after starting the server.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from api.schemas import ClassesResponse, HealthResponse, PredictionResponse, PredictionResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Global state populated at startup.
_state: dict = {}


HF_MODEL_REPO = "Stepang08/plant-disease-model"
HF_MODEL_FILE = "dinov2_best.pth"


def _download_model_if_needed() -> Path:
    """Download model weights from HuggingFace Hub if not present locally."""
    local_path = PROJECT_ROOT / "models" / "best_model.pth"
    if local_path.exists():
        return local_path
    print(f"Downloading model from {HF_MODEL_REPO}...")
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILE,
        local_dir=PROJECT_ROOT / "models",
    )
    dl_path = Path(downloaded)
    if dl_path.name != "best_model.pth":
        dl_path.rename(local_path)
    print(f"Model downloaded to {local_path}")
    return local_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at server start, release on shutdown."""
    from src.predict import load_inference_model

    try:
        checkpoint = _download_model_if_needed()
        model, label_names = load_inference_model(checkpoint)
        _state["model"] = model
        _state["label_names"] = label_names
        print(f"Model loaded: {checkpoint.name} ({len(label_names)} classes)")
    except Exception as e:
        print(f"WARNING: Failed to load model: {e}")
        _state["model"] = None
        _state["label_names"] = None
    yield
    _state.clear()


app = FastAPI(
    title="Plant Disease Detection API",
    description="Classifies plant diseases from leaf images. Outputs disease-only labels (e.g., 'late blight') regardless of host plant species.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Check if the service is running and the model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("model") is not None,
    )


@app.get("/classes", response_model=ClassesResponse)
def get_classes():
    """Return the list of disease classes the model can predict."""
    from src.predict import load_label_mapping

    names = load_label_mapping()
    return ClassesResponse(num_classes=len(names), classes=names)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="Plant/leaf image (JPG or PNG)")):
    """Upload a plant leaf image and get the predicted disease."""
    if _state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail=f"Expected an image file, got {content_type}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode the uploaded image.")

    from src.predict import predict_image

    import torch
    device = next(_state["model"].parameters()).device
    results = predict_image(
        model=_state["model"],
        image=image,
        label_names=_state["label_names"],
        device=device,
        top_k=5,
    )

    return PredictionResponse(
        disease=results[0]["disease"],
        confidence=results[0]["confidence"],
        top_k=[PredictionResult(**r) for r in results],
    )
