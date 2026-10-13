# ── Stage: runtime ──────────────────────────────
FROM python:3.11-slim

# System deps for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and frontend
COPY app/ ./app/
COPY frontend/ ./frontend/

# Copy model directory structure
COPY saved_models/ ./saved_models/

# TF-Hub cache directory
ENV TFHUB_CACHE_DIR=/app/.tfhub_modules
ENV PORT=8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
