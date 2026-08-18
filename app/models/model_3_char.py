"""
Model 3 — Character-level Conv1D
Architecture: CharVectorizer(290) → Embedding(32) → Conv1D(64)×2 → GlobalAvgPool → Dense(128) → Dense(5)
Saved artefact: saved_models/model_3.keras

To switch from mock to real:
  1. In the notebook: model_3.save('saved_models/model_3.keras')
  2. Place model_3.keras in saved_models/ and restart.
"""
import os
import logging
from pathlib import Path

from .base import BaseMedModel, SentenceResult, LABELS
from .mock import MockPredictor

logger = logging.getLogger(__name__)
WEIGHTS_PATH = Path(os.getenv("SAVED_MODELS_DIR", "saved_models")) / "model_3.keras"


def _split_chars(text: str) -> str:
    """Replicate notebook char-split: join individual chars with spaces."""
    return " ".join(list(text))


class Model3Char(BaseMedModel):
    MODEL_ID = "model_3"
    MODEL_NAME = "Char-level Conv1D"
    DESCRIPTION = "Character-level tokenisation fed into stacked 1-D convolutions."
    ARCHITECTURE = "CharVectorizer → Embedding(32) → Conv1D(64)×2 → GlobalAvgPool → Dense(128) → Dense(5)"
    ACCURACY = 66.4
    PRECISION = 67.9
    RECALL = 66.4
    F1_SCORE = 66.7
    TRAINING_TIME = "8 minutes"
    TRAINED_EPOCHS = "5 epochs"
    STRENGTH = "Robust against spelling errors/typos; solid baseline for morphological patterns."

    def __init__(self):
        super().__init__()
        self._model = None
        self._mock = MockPredictor()
        self.load()

    def load(self) -> bool:
        if WEIGHTS_PATH.exists():
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

        import tensorflow as tf

        # Model expects shape (batch, 1) string inputs (char-split)
        char_inputs = tf.constant([[_split_chars(s)] for s in sentences])
        proba = self._model.predict(char_inputs, verbose=0)  # (N, 5)
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
