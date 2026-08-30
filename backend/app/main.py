"""FastAPI application serving plant disease predictions."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import ALLOWED_CONTENT_TYPES, settings
from backend.app.model import PlantDiseaseClassifier
from backend.app.schemas import ErrorResponse, HealthResponse, PredictResponse, PredictionItem

logger = logging.getLogger(__name__)
classifier: PlantDiseaseClassifier | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup."""
    global classifier
    if not settings.model_path.exists():
        raise RuntimeError(f"Model checkpoint not found: {settings.model_path}")
    if not settings.labels_path.exists():
        raise RuntimeError(f"Label metadata not found: {settings.labels_path}")

    classifier = PlantDiseaseClassifier(settings.model_path, settings.labels_path)
    logger.info("Loaded model with %s classes on %s", len(classifier.class_names), classifier.device)
    yield
    classifier = None


app = FastAPI(
    title="Plant Disease Live Demo",
    description="REST API for crop leaf disease classification from smartphone photos.",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None,
        num_classes=len(classifier.class_names) if classifier else 0,
    )


@app.post("/api/v1/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...)) -> PredictResponse:
    """Classify an uploaded leaf image and return top-k predictions."""
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded", headers={"X-Error-Code": "MODEL_UNAVAILABLE"})

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {image.content_type}. Use JPEG, PNG, or WebP.",
            headers={"X-Error-Code": "INVALID_CONTENT_TYPE"},
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.", headers={"X-Error-Code": "EMPTY_FILE"})
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_bytes // (1024 * 1024)} MB limit.",
            headers={"X-Error-Code": "FILE_TOO_LARGE"},
        )

    try:
        predictions, inference_ms = classifier.predict(image_bytes, top_k=settings.top_k)
    except Exception as exc:  # noqa: BLE001 - surface safe message to client
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {exc}",
            headers={"X-Error-Code": "IMAGE_PROCESSING_FAILED"},
        ) from exc

    items = [PredictionItem(**prediction) for prediction in predictions]
    return PredictResponse(primary=items[0], top_k=items, inference_ms=round(inference_ms, 2))


@app.get("/")
async def root() -> FileResponse:
    """Serve the browser demo UI."""
    index_path = settings.frontend_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


if settings.frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")
