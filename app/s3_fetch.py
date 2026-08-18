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

    bucket_name = os.environ.get("S3_MODEL_BUCKET")
    if not bucket_name:
        logger.info("S3_MODEL_BUCKET not set. Skipping S3 model fetch.")
        return

    models_dir = Path(os.environ.get("SAVED_MODELS_DIR", "saved_models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    expected_models = [
        "model_1.keras",
        "model_2.keras",
        "model_3.keras",
        "model_4.keras",
        "model_5.keras",
    ]

    try:
        s3 = boto3.client('s3')
        logger.info(f"Checking S3 bucket '{bucket_name}' for model weights...")

        for model_name in expected_models:
            local_path = models_dir / model_name
            if not local_path.exists():
                logger.info(f"Downloading {model_name} from S3...")
                s3.download_file(bucket_name, f"models/{model_name}", str(local_path))
                logger.info(f"Successfully downloaded {model_name}.")
            else:
                logger.debug(f"{model_name} already exists locally. Skipping download.")

    except (NoCredentialsError, ClientError) as e:
        logger.error(f"S3 fetch failed: {e}. The API will automatically fall back to the Mock Model strategy for missing weights.")
    except Exception as e:
        logger.error(f"Unexpected error during S3 fetch: {e}. Falling back to Mock Model strategy.")
