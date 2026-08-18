# Changelog

All notable changes to MedSectAI are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [2.0.0] — 2026-08-19

### Added
- **Decoupled architecture:** Backend is now a headless JSON API (FastAPI on Hugging Face Spaces); frontend is a standalone static SPA (Vercel).
- **AWS S3 weight fetching:** Model weights are downloaded from S3 on backend startup via `app/s3_fetch.py`. Falls back to mock model if S3 is unavailable.
- **CORS middleware:** Backend now serves any origin, enabling the decoupled Vercel frontend to communicate freely.
- **`/models` endpoint:** Returns live metadata for all registered models.
- **`/debug` endpoint:** Returns filesystem and registry diagnostics for production troubleshooting.
- **`/compare` endpoint:** Runs all 6 models simultaneously and returns results keyed by model ID.
- **Model Zoo page:** New tab in the frontend showing detailed per-model metrics, hyperparameters, and training curves.
- **Multi-model comparison grid:** Frontend feature allowing side-by-side comparison of all 6 classifiers with a consensus agreement indicator.
- **Interactive Sandbox:** Real-time abstract classification with one-click sample abstracts.
- **Production Dockerfile:** Clean `python:3.11-slim` base, exposes port 7860 for HF Spaces, health-check on `/models`.
- `boto3==1.43.74` added to `requirements.txt`.

### Changed
- **Frontend completely rebuilt** as a pure static SPA. Removed all Jinja2/server-side rendering.
- Backend root route (`/`) changed from serving HTML to returning a JSON health-check response.
- Model loading refactored to support Keras 3 serialization (legacy Keras 2 models use architecture reconstruction).
- Repository re-initialized with a clean Git history — all macOS artifacts (`.DS_Store`, `__MACOSX`) and legacy files purged.

### Removed
- `app/templates/` — Legacy Jinja2 HTML templates.
- `app/static/` — Legacy CSS served by FastAPI. CSS now lives in `frontend/style.css`.
- `jinja2` dependency removed from `requirements.txt`.
- Fake Gradio config script removed from frontend HTML.

### Fixed
- Corrupt `boto3` entry in `requirements.txt` (encoding issue from Windows line endings).
- CSS path broken after frontend decoupling — fixed with relative path `href="style.css"`.

---

## [1.0.0] — 2026-08-01

### Added
- Initial project: 6 NLP models trained on PubMed 20k RCT dataset.
- FastAPI backend serving both the UI and the prediction API (monolithic).
- Jinja2-templated frontend with model selector and results view.
- Models:
  - Model 0: Naive Bayes (TF-IDF)
  - Model 1: Token Conv1D
  - Model 2: MiniLM sentence embeddings
  - Model 3: Character-level Conv1D
  - Model 4: Token + Char Hybrid
  - Model 5: Quadra (token + char + line position + total lines)
- Training notebooks in `notebooks/`.
