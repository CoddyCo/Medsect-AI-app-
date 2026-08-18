# Deployment Guide

This document describes how to deploy MedSectAI in its split-deployment architecture:
- **Backend** → Hugging Face Docker Space
- **Frontend** → Vercel

---

## Prerequisites

- A [Hugging Face](https://huggingface.co/) account
- A [Vercel](https://vercel.com/) account
- A [GitHub](https://github.com/) repository with this code pushed to `master`
- AWS S3 bucket containing your trained model weights (`.pkl` / `.keras` files)

---

## Step 1 — Push to GitHub

If you haven't done so already, push the project to GitHub:

```bash
git remote add origin https://github.com/YOUR_USERNAME/medsectai.git
git push -u origin master
```

---

## Step 2 — Deploy Backend to Hugging Face Spaces

### 2a. Create a new Docker Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the details:
   - **Owner:** Your HF username or organization
   - **Space name:** `medsectai-api` (or your preferred name)
   - **License:** MIT
   - **SDK:** **Gradio**
   - **Hardware:** CPU Basic (free tier) is fine for testing
   - **Visibility:** Public
3. Click **Create Space**

### 2b. Configure secrets

In your new Space, go to **Settings → Variables and secrets** and add:

| Secret Name | Value | Notes |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | Your AWS key ID | From IAM → Security credentials |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret | From IAM → Security credentials |
| `S3_MODEL_BUCKET` | Your S3 bucket name | e.g. `medsectai-weights` |

> ⚠️ **Never commit these values to Git.** Only add them via the HF Secrets UI.

### 2c. Link GitHub repo and push

The Hugging Face Space has its own Git remote. Push your code directly:

```bash
# Add the HF remote (one-time setup)
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/medsectai-api

# Push
git push hf master
```

> **Tip:** If prompted, your Hugging Face password is your HF **Access Token** (Profile → Access Tokens → Create token with `write` permission).

### 2d. Monitor the build

After pushing, go to your Space's page and click the **Logs** tab. The Docker build will take 3–5 minutes. You will see:

```
====== STARTING MODEL REGISTRY ======
...
INFO:app.s3_fetch:Downloading model_0.pkl from S3...
INFO:app.s3_fetch:All models downloaded successfully.
====== ALL MODELS LOADED ======
INFO:     Application startup complete.
```

Your backend URL will be: `https://YOUR_HF_USERNAME-medsectai-api.hf.space`

Test it:
```bash
curl https://YOUR_HF_USERNAME-medsectai-api.hf.space/
# Should return: {"status":"online","message":"MedSectAI API is running."}
```

---

## Step 3 — Update Frontend API URL

Before deploying the frontend, you need to update it to point at your live HF Space URL.

Open `frontend/index.html` and find this line near the top of the `<script>` block:

```javascript
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://127.0.0.1:8001' 
    : 'https://replace-with-your-huggingface-space-url.hf.space';
```

Replace the placeholder with your actual HF Space URL:

```javascript
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://127.0.0.1:8001' 
    : 'https://YOUR_HF_USERNAME-medsectai-api.hf.space';
```

Commit and push this change:
```bash
git add frontend/index.html
git commit -m "Set production backend URL"
git push origin master
git push hf master
```

---

## Step 4 — Deploy Frontend to Vercel

### 4a. Import your GitHub repo

1. Go to [vercel.com/new](https://vercel.com/new)
2. Click **Import** on your GitHub repo
3. In the **Configure Project** screen:
   - **Root Directory:** `frontend` ← important!
   - **Framework Preset:** Other
   - Leave all build settings blank (this is pure static HTML)
4. Click **Deploy**

Vercel will deploy in ~30 seconds. Your frontend URL will be something like:
`https://medsectai.vercel.app`

### 4b. Set up a custom domain (optional)

In Vercel → Project Settings → Domains, add your custom domain and follow the DNS instructions.

---

## Step 5 — Verify End-to-End

1. Open your Vercel frontend URL in a browser
2. Click **"Short — Knee OA Study"** to load a sample abstract
3. Click **"Run Classifier"**
4. Verify that colored sentence blocks appear in the results panel

If you see results — **congratulations, your deployment is live!** 🎉

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| HF build fails at `pip install` | TensorFlow version incompatibility | Pin `tensorflow==2.15.0` in `requirements.txt` |
| `WARNING: S3_MODEL_BUCKET not set` | Secrets not configured | Add secrets in HF Space settings |
| Frontend shows "Prediction failed" | CORS or wrong API URL | Verify `API_BASE` in `frontend/index.html` |
| Models show as MOCK | S3 download failed | Check AWS credentials and bucket name/region |
| Port 7860 not responding | Container failed to start | Check HF Space **Logs** tab for errors |
