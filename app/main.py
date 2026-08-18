"""
MedSectAI — FastAPI application entry point.
"""
import os

import logging
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
    # Point 3: Models will be lazy-loaded by the Singleton registry on first request, 
    # or we can force load them here if needed.


_BASE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(_BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(_BASE / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────
# Sample abstracts for the sidebar
# ────────────────────────────────────────────────
SAMPLE_ABSTRACTS = [
    {
        "label": "Short — Knee OA Study",
        "text": (
            "To investigate the efficacy of 6 weeks of daily low-dose oral prednisolone in improving pain, mobility, and systemic low-grade inflammation in older adults with moderate to severe knee osteoarthritis.\n"
            "A total of 125 patients with primary knee OA were randomized 1:1; 63 received 7.5 mg/day of prednisolone and 62 received placebo for 6 weeks.\n"
            "There was a clinically relevant reduction in the intervention group compared to the placebo group for knee pain, physical function, PGA, and 6MWD at 6 weeks.\n"
            "Low-dose oral prednisolone had both a short-term and a longer sustained effect resulting in less knee pain, better physical function, and attenuation of systemic inflammation in older patients with knee OA."
        ),
    },
    {
        "label": "Medium — Emotional Eating",
        "text": (
            "Emotional eating is associated with overeating and the development of obesity.\n"
            "Yet, empirical evidence for individual differences in emotional eating and cognitive mechanisms that contribute to eating during sad mood remain equivocal.\n"
            "The aim of this study was to test if attention bias for food moderates the effect of self-reported emotional eating during sad mood on actual food intake.\n"
            "Participants (N=96) were randomly assigned to one of the two experimental mood induction conditions.\n"
            "Hierarchical multivariate regression modeling showed that self-reported emotional eating did not account for changes in attention allocation for food or food intake in either condition.\n"
            "The current findings show that self-reported emotional eating might not validly predict who overeats when sad, at least not in a laboratory setting with healthy women."
        ),
    },
    {
        "label": "Long — Pulmonary Hypertension",
        "text": (
            "Pulmonary arterial hypertension (PAH) is a devastating disease characterised by progressive elevation of pulmonary arterial pressure leading to right heart failure and death.\n"
            "Current therapies target three pathways: endothelin, nitric oxide, and prostacyclin.\n"
            "Sotatercept is a first-in-class activin signalling inhibitor that targets the transforming growth factor-beta pathway.\n"
            "We conducted a phase 3, randomised, double-blind, placebo-controlled trial at 79 sites in 21 countries.\n"
            "Adults with PAH who were receiving stable background therapy were randomly assigned 1:1 to receive subcutaneous sotatercept or placebo every 3 weeks.\n"
            "The primary endpoint was the change from baseline in 6-minute walk distance at week 24.\n"
            "Sotatercept significantly improved 6-minute walk distance compared with placebo (least-squares mean difference 34.4 m, 95% CI 33.0–35.8; p<0.0001).\n"
            "Sotatercept significantly improved 6-minute walk distance and clinical outcomes in patients with PAH receiving stable background therapy."
        ),
    },
]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "models": registry.info(), "samples": SAMPLE_ABSTRACTS},
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
