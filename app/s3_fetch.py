import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def download_weights_from_s3():
    """
    Downloads model weights from an AWS S3 bucket if they do not exist locally.
    Designed with a try/except block so that if S3 is unreachable (e.g., missing credentials
    or no internet), it fails gracefully and the app seamlessly falls back to Mock models.
    """
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
    except ImportError:
        logger.warning("boto3 is not installed. Skipping S3 model fetch.")
        return

    if os.environ.get("FORCE_MOCK", "").lower() in ("true", "1"):
        logger.info("[S3_FETCH] FORCE_MOCK enabled. Skipping remote weight download; operating in Cloud Demo Mock mode.")
        return

    bucket_name = os.environ.get("S3_MODEL_BUCKET")
    if not bucket_name:
        logger.info("[S3_FETCH] S3_MODEL_BUCKET not configured. Running in Cloud Demo profile with Mock Predictor fallback.")
        return

    models_dir = Path(os.environ.get("SAVED_MODELS_DIR", "saved_models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    expected_models = [
        "model_0.pkl",
        "model_1.keras",
        "model_2.keras",
        "model_3.keras",
        "model_4.keras",
        "model_5.keras",
    ]

    try:
        s3 = boto3.client('s3')
        logger.info(f"[S3_FETCH] Connecting to S3 bucket '{bucket_name}' for model artifacts...")

        for model_name in expected_models:
            local_path = models_dir / model_name
            # Don't download if already exists and is non-empty (>50KB)
            if not local_path.exists() or local_path.stat().st_size < 50000:
                logger.info(f"[S3_FETCH] Downloading {model_name} from s3://{bucket_name}/models/{model_name}...")
                s3.download_file(bucket_name, f"models/{model_name}", str(local_path))
                logger.info(f"[S3_FETCH] Successfully downloaded {model_name}.")
            else:
                logger.debug(f"[S3_FETCH] {model_name} already exists locally. Skipping download.")

    except (NoCredentialsError, ClientError) as e:
        logger.warning(f"[S3_FETCH] S3 connection / credentials unavailable: {e}. Graceful fallback active: using Mock strategy.")
    except Exception as e:
        logger.warning(f"[S3_FETCH] S3 fetch error: {e}. Graceful fallback active: using Mock strategy.")
