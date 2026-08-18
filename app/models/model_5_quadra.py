"""
Model 5 — Quadra Model (Token + Char + Line Number + Total Lines)
Architecture:
  token_model   : USE(512) → Lambda → Dense(32)
  char_model    : CharVec → Emb(32) → BiLSTM(16)   → (32)
  line_num_model: OHE(15)             → Dense(32)
  total_lines_m : OHE(20)             → Dense(32)
  Concat(128) → Dropout(0.2) → Dense(64, relu) → Dropout(0.2) → Dense(5, softmax)

Saved artefact: saved_models/model_5.keras

This is the best-performing model (~87 % accuracy).

To switch from mock to real:
  1. In the notebook:        model_path = os.path.join("saved_models", "model_5_clean.keras")
  2. Place model_5.keras in saved_models/ and restart.
"""
import os
import logging
from pathlib import Path

from .base import BaseMedModel, SentenceResult, LABELS
from .mock import MockPredictor

logger = logging.getLogger(__name__)
WEIGHTS_PATH = Path(os.getenv("SAVED_MODELS_DIR", "saved_models")) / "model_5.keras"

# OHE depth constants — must match notebook
LINE_NUM_DEPTH = 15
TOTAL_LINES_DEPTH = 20


def _split_chars(text: str) -> str:
    return " ".join(list(text))


class Model5Quadra(BaseMedModel):
    MODEL_ID = "model_5"
    MODEL_NAME = "Quadra Model"
    DESCRIPTION = (
        "State-of-the-art multimodal model combining custom token embeddings, "
        "character embeddings (BiLSTM), and positional features (line number + total lines)."
    )
    ARCHITECTURE = (
        "TokenConv1D + CharVec→Emb(32)→BiLSTM(32) + LineNumOHE(15)→Dense(64) + TotalLinesOHE(20)→Dense(64) "
        "→ Concat(256) → Dense(512) → Dense(256) → Dense(5)"
    )
    ACCURACY = 86.9
    PRECISION = 86.8
    RECALL = 86.9
    F1_SCORE = 86.8
    TRAINING_TIME = "15 minutes"
    TRAINED_EPOCHS = "5 epochs"
    STRENGTH = "State-of-the-art performance across all classes; outstanding on METHODS and RESULTS due to positional knowledge."

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
            loaded = False

            try:
                import tensorflow as tf
                self._model = tf.keras.models.load_model(
                    str(WEIGHTS_PATH), compile=False
                )
                logger.info(f"{self.MODEL_ID}: Loaded successfully with modern Keras 3.")
                loaded = True
            except Exception as e:
                logger.exception(f"{self.MODEL_ID}: Load failed")

            if not loaded:
                logger.warning(f"{self.MODEL_ID}: All load strategies failed - using mock")
                return self._mock.predict(sentences)

        import tensorflow as tf

        n = len(sentences)
        total = n  # use total sentence count as proxy for total_lines

        token_inputs      = tf.constant(sentences)
        char_inputs       = tf.constant([[_split_chars(s)] for s in sentences])
        line_num_ohe      = tf.one_hot([min(i, LINE_NUM_DEPTH - 1) for i in range(n)], depth=LINE_NUM_DEPTH)
        total_lines_ohe   = tf.one_hot([min(total, TOTAL_LINES_DEPTH - 1)] * n,       depth=TOTAL_LINES_DEPTH)

        proba     = self._model.predict(
            [token_inputs, char_inputs, line_num_ohe, total_lines_ohe], verbose=0
        )
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

