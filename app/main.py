"""
MedSectAI — FastAPI application entry point.
"""
import os

import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .model_registry import registry
from .schemas import PredictRequest, CompareRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MedSectAI",
    description="Medical Abstract Section Classifier API",
    version="1.0.0",
)

@app.on_event("startup")
async def startup_event():
    from .s3_fetch import download_weights_from_s3
    logger.info("Initializing application startup sequence...")
    # Point 1: Fetch weights from S3
    download_weights_from_s3()
@app.get("/health")
async def health():
    return {"status": "online", "message": "MedSectAI API is running.", "models": len(registry.all())}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(request: PredictRequest):
    import time
    model = registry.get(request.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found.")

    mode = "MOCK ⚠️" if model.is_mock else "REAL ✅"
    logger.info(
        "PREDICT → model=%s (%s) [%s]",
        model.MODEL_ID, model.MODEL_NAME, mode,
    )

    t0 = time.perf_counter()
    results = model.predict(request.text)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "PREDICT ← model=%s [%s] | %d sentences | %.1f ms",
        model.MODEL_ID, mode, len(results), elapsed_ms,
    )
    return {
        "model_id": model.MODEL_ID,
        "model_name": model.MODEL_NAME,
        "is_mock": model.is_mock,
        "predictions": [r.to_dict() for r in results],
    }


@app.post("/compare")
async def compare(request: CompareRequest):
    import time
    all_results = {}
    for m in registry.all():
        mode = "MOCK ⚠️" if m.is_mock else "REAL ✅"
        t0 = time.perf_counter()
        results = m.predict(request.text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "COMPARE ← model=%s [%s] | %d sentences | %.1f ms",
            m.MODEL_ID, mode, len(results), elapsed_ms,
        )
        all_results[m.MODEL_ID] = {
            "model_name": m.MODEL_NAME,
            "is_mock": m.is_mock,
            "predictions": [r.to_dict() for r in results],
        }
    return all_results


@app.get("/models")
async def list_models():
    return registry.info()


@app.get("/debug")
async def debug():
    import traceback
    import sys
    from pathlib import Path
    
    debug_info = {}
    models_dir = os.environ.get("SAVED_MODELS_DIR", "saved_models")
    debug_info["SAVED_MODELS_DIR"] = models_dir
    debug_info["dir_exists"] = Path(models_dir).exists()
    
    if debug_info["dir_exists"]:
        debug_info["files"] = {str(f.name): f.stat().st_size for f in Path(models_dir).iterdir()}
        
    debug_info["models"] = registry.info()
    return debug_info


# ── Mount static frontend (if directory present) ───────────────────────────
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    @app.get("/")
    async def index():
        return {"status": "online", "message": "MedSectAI API is running. Mount frontend to view UI."}
