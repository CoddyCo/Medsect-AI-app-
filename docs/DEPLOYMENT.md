# Deployment Guide — MedSectAI

MedSectAI is architected for cloud-native deployment with **production S3 weight fetching**, **graceful mock fallback**, and **containerized microservices**.

---

## 🎯 Architecture & Deployment Strategy

Running 6 deep learning models simultaneously (BiLSTM, Conv1D, Transformers) requires ~2–4GB of RAM and dedicated hardware. For portfolio and resume demonstrations, cloud providers (Render, Hugging Face, AWS) offer free tiers with 512MB–1GB RAM limits that crash under heavy deep-learning memory loads.

To solve this, MedSectAI features an **Enterprise Cloud Demo Profile**:
- **Production S3 Retrieval Engine (`app/s3_fetch.py`):** Automatically checks for AWS credentials and an S3 bucket to retrieve real model weights (`.keras` / `.pkl`).
- **Zero-Crash Mock Fallback Strategy:** If S3 credentials are not supplied, weights are missing, or `FORCE_MOCK=true` is set, the application gracefully activates lightweight mock predictors.
- **Benefits:**
  - Fast cold-start boot time (<3 seconds).
  - <70MB RAM consumption (runs flawlessly on free tiers forever).
  - Demonstrates full senior cloud architecture, S3 pipelines, Docker containerization, and interactive UI without incurring cloud bills.

---

## 🚀 Recommended: 1-Click Free Deployment on Render.com

[Render.com](https://render.com) offers a 100% free Web Service tier that connects directly to GitHub, builds the `Dockerfile`, and serves both the FastAPI API and the frontend UI.

### Step 1: Push Code to GitHub
Ensure your latest changes are pushed:
```bash
git push origin master
```

### Step 2: Create a Web Service on Render
1. Sign in to [Render.com](https://render.com) using your GitHub account.
2. Click **New +** → **Web Service**.
3. Under **Connect a repository**, find and select `Medsect-AI-app-`.
4. Configure service settings:
   - **Name:** `medsect-ai` (or your preferred name)
   - **Language:** `Docker` (automatically detected from `Dockerfile`)
   - **Region:** Any (e.g., Oregon or Frankfurt)
   - **Instance Type:** **Free** ($0 / month)
5. Under **Environment Variables** (Optional):
   - `FORCE_MOCK` = `true` (guarantees lightweight demo mode)
   - *(If you later wish to use real S3 weights, add `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `S3_MODEL_BUCKET` here).*
6. Click **Create Web Service**.

### Step 3: Access Your Live Application
Render will build the Docker container in ~2–3 minutes. Once the build finishes:
- **Full Interactive Web App:** `https://your-service-name.onrender.com/`
- **Interactive Swagger OpenAPI Docs:** `https://your-service-name.onrender.com/docs`
- **Health Check:** `https://your-service-name.onrender.com/health`

---

## 🌐 Alternative: Split Deployment (Frontend on Vercel)

If you wish to demonstrate decoupled microservices:

1. Deploy the backend to Render (as described above).
2. Deploy the frontend to Vercel:
   - Go to [vercel.com/new](https://vercel.com/new) and import `Medsect-AI-app-`.
   - Set **Root Directory** to `frontend`.
   - In `frontend/index.html`, set `window.API_BASE_URL = 'https://your-service-name.onrender.com'`.
   - Click **Deploy**.

---

## 🔍 Verification & Health Check

After deployment, test your live endpoint:

```bash
# Health check
curl https://your-service-name.onrender.com/health

# Predict endpoint
curl -X POST https://your-service-name.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "A randomized trial was conducted to evaluate drug efficacy.", "model_id": "model_0"}'
```
