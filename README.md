<div align="center">

# 🧠 MedSectAI

### Automatic Medical Abstract Section Classifier

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com)
[![Hugging Face Spaces](https://img.shields.io/badge/Backend-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces)

**MedSectAI** automatically segments unstructured medical abstracts into clear, standardized clinical sections — *Background, Objective, Methods, Results,* and *Conclusions* — using 6 purpose-trained NLP pipelines.

[**Live Demo →**](https://medsectai.vercel.app) · [**API Docs →**](https://huggingface.co/spaces/Mr-underscore/medsectai-api/docs) · [**Report a Bug**](issues) · [**Request a Feature**](issues)

---

![MedSectAI Dashboard](https://raw.githubusercontent.com/CoddyCo/Medsect-AI-app/main/docs/assets/dashboard-preview.png)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| **6 NLP Pipelines** | From a Naive Bayes baseline to a state-of-the-art Quadra multimodal model |
| **88% F1Score** | Best-in-class performance with the Quadra model on 20,000 PubMed RCT papers |
| **Decoupled Architecture** | FastAPI backend on Hugging Face, static frontend on Vercel |
| **Cloud Weight Fetching** | Model weights are fetched from AWS S3 on startup — zero weight bloat in the repo |
| **Interactive Sandbox** | Paste any abstract and see it classified sentence-by-sentence in real-time |
| **Multi-Model Comparison** | Run all 6 classifiers simultaneously and inspect consensus agreement |
| **< 12ms Inference** | Highly optimized inference pipeline with sub-12ms response latency |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User's Browser                      │
│                                                         │
│   ┌───────────────────────────────────────────────┐    │
│   │         Frontend (Vercel CDN)                  │    │
│   │    HTML + Vanilla CSS + JavaScript            │    │
│   │    ✦ Dashboard   ✦ Sandbox   ✦ Model Zoo      │    │
│   └─────────────────────┬─────────────────────────┘    │
└─────────────────────────┼───────────────────────────────┘
                          │  HTTPS JSON (REST API)
                          ▼
┌─────────────────────────────────────────────────────────┐
│          Backend (Hugging Face Docker Space)            │
│                                                         │
│   ┌───────────────────────────────────────────────┐    │
│   │         FastAPI (Uvicorn, Port 7860)           │    │
│   │  POST /predict    POST /compare    GET /models │    │
│   └───────────────────┬───────────────────────────┘    │
│                       │                                 │
│   ┌───────────────────▼───────────────────────────┐    │
│   │             Model Registry                     │    │
│   │  Model 0 · Model 1 · Model 2 · Model 3         │    │
│   │  Model 4 · Model 5 (Quadra)                    │    │
│   └───────────────────┬───────────────────────────┘    │
│                       │ On Startup                      │
│   ┌───────────────────▼───────────────────────────┐    │
│   │          S3 Weight Fetcher (boto3)             │    │
│   │      Downloads .keras / .pkl weights           │    │
│   └───────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │    AWS S3 Bucket      │
              │  model_0.pkl          │
              │  model_1.keras        │
              │  model_2.keras        │
              │  model_3.keras        │
              │  model_4.keras        │
              │  model_5.keras        │
              └───────────────────────┘
```

---

## 📊 Model Zoo

Six progressively complex NLP classifiers were trained on the [PubMed 20k RCT dataset](https://github.com/Franck-Dernoncourt/pubmed-rct).

| # | Model Name | Architecture | Accuracy | F1-Score | Weights |
|---|---|---|---|---|---|
| 0 | **Naive Bayes** | TF-IDF → Multinomial NB | 72.2% | 69.8% | `model_0.pkl` |
| 1 | **Token Conv1D** | Embedding(128) → Conv1D(64) → Dense | 80.1% | 79.8% | `model_1.keras` |
| 2 | **MiniLM** | `all-MiniLM-L6-v2` (384-d) → Dense | 76.4% | 75.9% | `model_2.keras` |
| 3 | **Char Conv1D** | Char Embedding → BiLSTM → Dense | 66.4% | 66.7% | `model_3.keras` |
| 4 | **Hybrid** | Token + Char → Concat → Dense | 80.6% | 80.4% | `model_4.keras` |
| 5 | **Quadra** 🏆 | Token + Char + Line Pos + Total Lines | **86.9%** | **86.8%** | `model_5.keras` |

> Validation split: 20k PubMed RCT abstracts. All models trained for 5 epochs.

---

## 🚀 Quick Start — Local Development

### Prerequisites
- Python 3.11+
- Git

### 1. Clone & set up the environment

```bash
git clone https://github.com/CoddyCo/Medsect-AI-app-.git
cd Medsect-AI-app-

# Create and activate virtual environment
python -m venv mle
# Windows:
.\mle\Scripts\activate
# macOS/Linux:
source mle/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add model weights

Place your trained model files in the `saved_models/` directory:
```
saved_models/
├── model_0.pkl
├── model_1.keras
├── model_2.keras
├── model_3.keras
├── model_4.keras
└── model_5.keras
```

> **Note:** Model weights are not tracked in Git (see `.gitignore`). They are fetched from AWS S3 on the deployed backend automatically.

### 3. Start the backend

```bash
.\mle\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8001 --reload
```

The API will be available at `http://127.0.0.1:8001`.
Interactive API docs (Swagger UI) at `http://127.0.0.1:8001/docs`.

### 4. Serve the frontend

In a separate terminal:

```bash
cd frontend
python -m http.server 3000
```

Open `http://localhost:3000` in your browser.

---

## 🌐 API Reference

Base URL (production): `https://CoddyCo-medsectai-api.hf.space`

### `GET /`

Health check.

**Response:**
```json
{
  "status": "online",
  "message": "MedSectAI API is running."
}
```

---

### `POST /predict`

Classify a medical abstract using a single model.

**Request body:**
```json
{
  "text": "Patients were randomized into two groups...\nThe primary endpoint was 6-minute walk distance.",
  "model_id": "model_5"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `text` | `string` | ✅ | — | The abstract text. Newline-separated or paragraph form. |
| `model_id` | `string` | ❌ | `"model_0"` | One of `model_0` … `model_5` |

**Response:**
```json
{
  "model_id": "model_5",
  "model_name": "Quadra Model",
  "is_mock": false,
  "predictions": [
    {
      "sentence": "Patients were randomized into two groups...",
      "label": "METHODS",
      "confidence": 94.7,
      "line_number": 1
    }
  ]
}
```

---

### `POST /compare`

Run all 6 models simultaneously and return their predictions side-by-side.

**Request body:**
```json
{
  "text": "The aim of this study was to evaluate..."
}
```

**Response:** A dictionary keyed by `model_id`, each value following the same structure as `/predict`.

---

### `GET /models`

Returns metadata for all registered models.

**Response:**
```json
[
  {
    "id": "model_0",
    "name": "Naive Bayes (TF-IDF)",
    "architecture": "TF-IDF + Multinomial Naive Bayes",
    "accuracy": 72.2,
    "is_mock": false
  },
  ...
]
```

---

## ☁️ Deployment

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the complete, step-by-step deployment guide to Hugging Face Spaces (backend) and Vercel (frontend).

---

## 📁 Project Structure

```
medsectai/
├── app/                        # FastAPI backend
│   ├── main.py                 # App entry point, routes
│   ├── model_registry.py       # Singleton model registry
│   ├── s3_fetch.py             # AWS S3 weight downloader
│   ├── schemas.py              # Pydantic request/response schemas
│   └── models/                 # Per-model loading & inference logic
│       ├── base.py             # Abstract base class for all models
│       ├── mock.py             # Mock model (fallback if weights missing)
│       ├── model_0_naive.py    # Naive Bayes (sklearn)
│       ├── model_1_conv1d.py   # Token Conv1D (Keras)
│       ├── model_2_minilm.py   # MiniLM sentence embeddings
│       ├── model_3_char.py     # Character-level Conv1D
│       ├── model_4_hybrid.py   # Token + Char hybrid
│       └── model_5_quadra.py   # Quadra multimodal (best model)
│
├── frontend/                   # Static frontend (Vercel)
│   ├── index.html              # Single-page app
│   └── style.css               # All styles
│
├── notebooks/                  # Training & experimentation notebooks
│   ├── MedSectAI.ipynb         # Main training notebook
│   └── MedSectAI_optm.ipynb    # Optimized training
│
├── scripts/
│   └── train_models.py         # Standalone training script
│
├── saved_models/               # ⚠️ Git-ignored — fetch from S3
│
├── Dockerfile                  # Docker image for HF Spaces
├── docker-compose.yml          # Local Docker development
├── requirements.txt            # Python dependencies
└── .gitignore
```

---

## 🤝 Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgements

- [PubMed 20k RCT Dataset](https://github.com/Franck-Dernoncourt/pubmed-rct) — Dernoncourt & Lee (2017)
- [Sentence-Transformers](https://www.sbert.net/) — Reimers & Gurevych (2019)
- [FastAPI](https://fastapi.tiangolo.com/) — Sebastián Ramírez

