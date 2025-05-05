import os
import sys
import time
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Body, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import cv2
from PIL import Image
import io

# Ultralytics YOLOv8
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('api')

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "weights/synthetic_only.pt")
DATASET_PATH = os.environ.get("DATASET_PATH", "dataset")
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.json")

app = FastAPI(
    title="Istanbul Waste Detection API",
    description="API for detecting waste in drone imagery using synthetic-trained YOLOv8",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model cache
model_cache = {}

# Pydantic models for API
class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float] = Field(..., description="Bounding box in [x, y, width, height] format")

class PredictionResponse(BaseModel):
    predictions: List[DetectionResult]
    inference_time: float
    model_name: str
    image_size: List[int]
    timestamp: str

class ModelInfo(BaseModel):
    name: str
    path: str
    loaded: bool
    classes: List[str]
    resolution: List[int]

class DatasetStats(BaseModel):
    num_images: int
    num_classes: int
    classes: List[str]
    environment_distribution: Dict[str, int]
    lighting_distribution: Dict[str, int]
    class_distribution: Dict[str, int]

def load_metadata():
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return None

def load_model(model_path: str = DEFAULT_MODEL_PATH):
    if model_path in model_cache:
        return model_cache[model_path]

    try:
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        model_cache[model_path] = model
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

def get_model(model_path: str = Query(DEFAULT_MODEL_PATH, description="Path to model weights")):
    return load_model(model_path)

# Routes
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Istanbul Waste Detection API",
        "version": "1.0.0",
        "description": "API for detecting waste in drone imagery using synthetic-trained YOLOv8",
        "endpoints": {
            "/predict": "POST - Detect waste in uploaded image",
            "/models": "GET - List available models",
            "/dataset": "GET - Get dataset statistics",
            "/health": "GET - Check API health"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model: YOLO = Depends(get_model),
    conf_threshold: float = Query(0.25, ge=0.01, le=1.0, description="Confidence threshold"),
    iou_threshold: float = Query(0.45, ge=0.01, le=1.0, description="IoU threshold")
):
    start_time = time.time()

    try:
        # Read image
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        height, width = image.shape[:2]

        results = model(image, conf=conf_threshold, iou=iou_threshold)[0]

        predictions = []

        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection

            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            class_name = results.names[int(class_id)]

            predictions.append(DetectionResult(
                class_id=int(class_id),
                class_name=class_name,
                confidence=float(confidence),
                bbox=[float(x), float(y), float(w), float(h)]
            ))

        predictions.sort(key=lambda x: x.confidence, reverse=True)

        inference_time = time.time() - start_time

        return PredictionResponse(
            predictions=predictions,
            inference_time=inference_time,
            model_name=Path(DEFAULT_MODEL_PATH).stem,
            image_size=[width, height],
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up
        await file.close()

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models."""
    try:
        weights_dir = Path("weights")
        if not weights_dir.exists():
            return []

        models = []
        for model_path in weights_dir.glob("*.pt"):
            is_loaded = str(model_path) in model_cache

            # Get class names from loaded model or metadata
            classes = []
            resolution = [1280, 720]  # Default

            if is_loaded:
                model = model_cache[str(model_path)]
                classes = list(model.names.values())
            else:
                metadata = load_metadata()
                if metadata:
                    classes = metadata["dataset_info"]["classes"]
                    resolution = metadata["dataset_info"]["resolution"]

            models.append(ModelInfo(
                name=model_path.stem,
                path=str(model_path),
                loaded=is_loaded,
                classes=classes,
                resolution=resolution
            ))

        return models
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.get("/dataset", response_model=DatasetStats)
async def get_dataset_stats():
    """Get dataset statistics."""
    try:
        metadata = load_metadata()
        if not metadata:
            raise HTTPException(status_code=404, detail="Metadata not found")

        return DatasetStats(
            num_images=metadata["dataset_info"]["num_images"],
            num_classes=len(metadata["dataset_info"]["classes"]),
            classes=metadata["dataset_info"]["classes"],
            environment_distribution=metadata["statistics"]["environment_distribution"],
            lighting_distribution=metadata["statistics"]["lighting_distribution"],
            class_distribution=metadata["statistics"]["class_distribution"]
        )
    except Exception as e:
        logger.error(f"Error getting dataset stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting dataset stats: {str(e)}")

@app.get("/health")
async def health_check():
    """Check API health."""
    try:
        load_model()

        metadata = load_metadata()
        if not metadata:
            return {"status": "warning", "message": "Metadata not found, but API is operational"}

        return {"status": "healthy", "version": "1.0.0"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

try:
    app.mount("/static", StaticFiles(directory="api/static"), name="static")

    @app.get("/demo", include_in_schema=False)
    async def demo():
        """Demo UI for testing the API."""
        return FileResponse("api/static/index.html")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

if __name__ == "__main__":
    import uvicorn

    load_model()

    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("DEBUG", "").lower() == "true"
    )
