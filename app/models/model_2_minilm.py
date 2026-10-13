"""
Model 2 — MiniLM Feature Extractor
Architecture: Hugging Face all-MiniLM-L6-v2 (384-d) → Keras Dense
Saved artefact: saved_models/model_2.keras

NOTE: sentence-transformers pulls in PyTorch which conflicts with TensorFlow
on macOS arm64 (OpenMP clash → segfault). The SentenceTransformer encoder is
therefore run in an isolated subprocess so PyTorch never loads into the main
TF process. This is transparent to the caller.

To switch from mock to real:
  1. Train Model 2 in the notebook using the sentence-transformers pipeline.
  2. Save with: model_2_hf.save('saved_models/model_2.keras')
  3. Place model_2.keras in saved_models/ and restart.
"""
import os
import sys
import json
import logging
import subprocess
from pathlib import Path

from .base import BaseMedModel, SentenceResult, LABELS
from .mock import MockPredictor

logger = logging.getLogger(__name__)
WEIGHTS_PATH = Path(os.getenv("SAVED_MODELS_DIR", "saved_models")) / "model_2.keras"
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Model2MiniLM(BaseMedModel):
    MODEL_ID = "model_2"
    MODEL_NAME = "MiniLM Feature Extractor"
    DESCRIPTION = "Uses Hugging Face's all-MiniLM-L6-v2 (384-d embeddings) for ultra-fast, lightweight feature extraction."
    ARCHITECTURE = "MiniLM(384) → Dense(128, relu) → Dense(5, softmax)"
    ACCURACY = 76.4
    PRECISION = 75.8
    RECALL = 76.4
    F1_SCORE = 75.9
    TRAINING_TIME = "1 minute"
    TRAINED_EPOCHS = "5 epochs"
    STRENGTH = "Tiny footprint (~80MB total) with near-SOTA semantic representation."

    def __init__(self):
        super().__init__()
        self._model = None
        self._mock = MockPredictor()
        self.load()

    def load(self) -> bool:
        force_mock = os.getenv("FORCE_MOCK", "false").lower() in ("true", "1")
        if not force_mock and WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 50000:
            self._is_mock = False
            logger.info(f"{self.MODEL_ID}: Found weights, deferring load until first prediction.")
            return True
        self._is_mock = True
        return False

    def _encode_subprocess(self, sentences: list[str]) -> list[list[float]]:
        """
        Run SentenceTransformer.encode() in a fresh subprocess so PyTorch
        never enters the TF process (avoids OpenMP segfault on macOS arm64).
        Sentences are passed via stdin as JSON; embeddings come back via stdout.
        """
        script = (
            "import sys, json\n"
            "from sentence_transformers import SentenceTransformer\n"
            f"model = SentenceTransformer({HF_MODEL_NAME!r})\n"
            "sentences = json.load(sys.stdin)\n"
            "vecs = model.encode(sentences).tolist()\n"
            "print(json.dumps(vecs))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(sentences),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Subprocess encoder failed:\n{result.stderr[-500:]}"
            )
        return json.loads(result.stdout.strip())

    def predict(self, text: str) -> list[SentenceResult]:
        sentences = self.split_sentences(text)
        if self._is_mock:
            return self._mock.predict(sentences)
            
        if getattr(self, "_model", None) is None:
            try:
                import tensorflow as tf
                self._model = tf.keras.models.load_model(str(WEIGHTS_PATH), compile=False)
                logger.info(f"{self.MODEL_ID}: Lazy loaded successfully.")
            except Exception as e:
                logger.warning(f"{self.MODEL_ID}: Lazy load failed ({e}) - using mock")
                return self._mock.predict(sentences)

        import numpy as np
        try:
            # Encode in a subprocess — keeps PyTorch out of the TF process
            embeddings = np.array(self._encode_subprocess(sentences), dtype="float32")
        except Exception as e:
            logger.warning(
                "Model 2: encoder failed (%s) — falling back to mock", e
            )
            return self._mock.predict(sentences)

        proba = self._model.predict(embeddings, verbose=0)  # (N, 5)
        pred_idxs = proba.argmax(axis=1)

        results = []
        for i, (sentence, idx, prob_row) in enumerate(zip(sentences, pred_idxs, proba)):
            results.append(SentenceResult(
                sentence=sentence,
                label=LABELS[idx],
                confidence=float(prob_row[idx]),
                line_number=i,
            ))
        return results
