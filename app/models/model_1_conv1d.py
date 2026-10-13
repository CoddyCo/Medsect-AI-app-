"""
Model 1 — Token-level Conv1D
Architecture: TextVectorization → Embedding(128) → Conv1D(64) → GlobalAvgPool → Dense(5)
Saved artefact: saved_models/model_1.keras

To switch from mock to real:
  1. In the notebook: model_1.save('saved_models/model_1.keras')
  2. Place model_1.keras in saved_models/ and restart.
"""
import os
import logging
from pathlib import Path

from .base import BaseMedModel, SentenceResult, LABELS
from .mock import MockPredictor

logger = logging.getLogger(__name__)
WEIGHTS_PATH = Path(os.getenv("SAVED_MODELS_DIR", "saved_models")) / "model_1.keras"


class Model1Conv1D(BaseMedModel):
    MODEL_ID = "model_1"
    MODEL_NAME = "Token Conv1D"
    DESCRIPTION = "Token-level embeddings processed with a 1-D convolutional neural network."
    ARCHITECTURE = "TextVectorization → Embedding(128) → Conv1D(64) → GlobalAvgPool → Dense(5)"
    ACCURACY = 80.1
    PRECISION = 79.8
    RECALL = 80.1
    F1_SCORE = 79.8
    TRAINING_TIME = "4 minutes"
    TRAINED_EPOCHS = "5 epochs"
    STRENGTH = "Good balanced token representation; high recall on METHODS."

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
        import tensorflow as tf

        # Model expects shape (batch, 1) string inputs
        inputs = tf.constant([[s] for s in sentences])
        proba = self._model.predict(inputs, verbose=0)  # (N, 5)
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
