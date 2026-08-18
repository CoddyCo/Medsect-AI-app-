"""
Model 0 — Naive Baseline
Architecture: sklearn Pipeline (TF-IDF → Multinomial Naive Bayes)
Saved artefact: saved_models/model_0.pkl

To switch from mock to real:
  1. Train the model in the notebook.
  2. Save with: import pickle; pickle.dump(model_0, open('saved_models/model_0.pkl','wb'))
  3. Place model_0.pkl in saved_models/ and restart the container.
"""
import os
import logging
from pathlib import Path

from .base import BaseMedModel, SentenceResult, LABELS
from .mock import MockPredictor

logger = logging.getLogger(__name__)
WEIGHTS_PATH = Path(os.getenv("SAVED_MODELS_DIR", "saved_models")) / "model_0.pkl"


class Model0Naive(BaseMedModel):
    MODEL_ID = "model_0"
    MODEL_NAME = "Naive Bayes (TF-IDF)"
    DESCRIPTION = "Baseline model using TF-IDF vectorization with Multinomial Naive Bayes classifier."
    ARCHITECTURE = "TF-IDF + Multinomial Naive Bayes (sklearn Pipeline)"
    ACCURACY = 72.2
    PRECISION = 71.9
    RECALL = 72.2
    F1_SCORE = 69.8
    TRAINING_TIME = "15 seconds"
    TRAINED_EPOCHS = "N/A"
    STRENGTH = "Fastest training & prediction; high precision on RESULTS."

    def __init__(self):
        super().__init__()
        self._pipeline = None
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

        if getattr(self, "_pipeline", None) is None:
            try:
                import pickle
                with open(WEIGHTS_PATH, "rb") as f:
                    self._pipeline = pickle.load(f)
                logger.info(f"{self.MODEL_ID}: Lazy loaded successfully.")
            except Exception as e:
                logger.warning(f"{self.MODEL_ID}: Lazy load failed ({e}) - using mock")
                return self._mock.predict(sentences)

        # Real inference
        # pipeline.predict() returns the string class label directly
        # (e.g. "BACKGROUND"), NOT an integer index.
        preds = self._pipeline.predict(sentences)
        proba = self._pipeline.predict_proba(sentences)

        results = []
        for i, (sentence, label, prob_row) in enumerate(
            zip(sentences, preds, proba)
        ):
            confidence = float(prob_row.max())
            results.append(SentenceResult(
                sentence=sentence,
                label=str(label),
                confidence=confidence,
                line_number=i,
            ))
        return results
