"""
Abstract base class for all MedSectAI models.

To add a new model:
  1. Create a new file in app/models/
  2. Subclass BaseModel
  3. Implement load() and predict()
  4. Register in app/model_registry.py
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


LABELS = ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"]

# Color mapping consumed by the template
LABEL_COLORS = {
    "BACKGROUND":  "#3b82f6",   # blue
    "METHODS":     "#22c55e",   # green
    "RESULTS":     "#f97316",   # orange
    "CONCLUSIONS": "#ef4444",   # red
    "OBJECTIVE":   "#a855f7",   # purple
}


@dataclass
class SentenceResult:
    sentence: str
    label: str
    confidence: float
    line_number: int
    color: str = field(init=False)

    def __post_init__(self):
        self.color = LABEL_COLORS.get(self.label, "#6b7280")

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "label": self.label,
            "confidence": round(self.confidence * 100, 1),
            "line_number": self.line_number,
            "color": self.color,
        }


class BaseMedModel(ABC):
    """Abstract base for all section classifiers."""

    #: Override in subclass
    MODEL_ID: str = ""
    MODEL_NAME: str = ""
    DESCRIPTION: str = ""
    ARCHITECTURE: str = ""
    ACCURACY: Optional[float] = None
    PRECISION: Optional[float] = None
    RECALL: Optional[float] = None
    F1_SCORE: Optional[float] = None
    TRAINING_TIME: Optional[str] = None
    TRAINED_EPOCHS: Optional[str] = None
    STRENGTH: Optional[str] = None

    def __init__(self):
        self._is_loaded = False
        self._is_mock = True

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    @abstractmethod
    def load(self) -> bool:
        """
         Attempt to load weights / artefacts.
         Return True if real model loaded, False to stay in mock mode.
         """

    @abstractmethod
    def predict(self, text: str) -> list[SentenceResult]:
        """
         Given a raw abstract text (may be multi-sentence),
         return a SentenceResult for each sentence.
         """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Split abstract into sentences (one per line, or period-split)."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            # Fall back to naive period split
            import re
            lines = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        return lines

    def info(self) -> dict:
        return {
            "id": self.MODEL_ID,
            "name": self.MODEL_NAME,
            "description": self.DESCRIPTION,
            "architecture": self.ARCHITECTURE,
            "accuracy": self.ACCURACY,
            "precision": self.PRECISION,
            "recall": self.RECALL,
            "f1_score": self.F1_SCORE,
            "training_time": self.TRAINING_TIME,
            "trained_epochs": self.TRAINED_EPOCHS,
            "strength": self.STRENGTH,
            "is_mock": self.is_mock,
        }
