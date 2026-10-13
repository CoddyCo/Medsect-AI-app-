"""
Model 4 — Token + Char Hybrid (no positional embedding)
Architecture:
  Token branch : TF-Hub USE(512) → Lambda → Dense(32)
  Char branch  : CharVectorizer → Embedding(32) → Bidirectional LSTM(16)
  Concat → Dense(5, softmax)
Saved artefact: saved_models/model_4.keras

To switch from mock to real:
  1. In the notebook: model_4.save('saved_models/model_4.keras')
  2. Place model_4.keras in saved_models/ and restart.
"""
import os
import logging
from pathlib import Path

from .base import BaseMedModel, SentenceResult, LABELS
from .mock import MockPredictor

logger = logging.getLogger(__name__)
WEIGHTS_PATH = Path(os.getenv("SAVED_MODELS_DIR", "saved_models")) / "model_4.keras"


def _split_chars(text: str) -> str:
    return " ".join(list(text))


class Model4Hybrid(BaseMedModel):
    MODEL_ID = "model_4"
    MODEL_NAME = "Token + Char Hybrid"
    DESCRIPTION = "Combines custom token embeddings with character-level Bidirectional LSTM representations."
    ARCHITECTURE = "TokenConv1D + CharVec→Emb(32)→BiLSTM(32) → Concat → Stacked Dense → Dense(5)"
    ACCURACY = 80.6
    PRECISION = 80.5
    RECALL = 80.6
    F1_SCORE = 80.4
    TRAINING_TIME = "12 minutes"
    TRAINED_EPOCHS = "5 epochs"
    STRENGTH = "Blends semantic word context and character morphology; high precision on OBJECTIVE."

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

        import tensorflow as tf

        token_inputs = tf.constant(sentences)
        char_inputs = tf.constant([[_split_chars(s)] for s in sentences])

        proba = self._model.predict([token_inputs, char_inputs], verbose=0)  # (N, 5)
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
