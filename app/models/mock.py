"""
Shared mock predictor — returns random labels when real weights are missing.
Used by every model as a drop-in fallback.
"""
import random
from .base import BaseMedModel, SentenceResult, LABELS


class MockPredictor:
    """Deterministic-ish random predictor for UI testing."""

    def predict(self, sentences: list[str]) -> list[SentenceResult]:
        results = []
        for i, sentence in enumerate(sentences):
            # Seed on sentence so results are stable within a session
            rng = random.Random(hash(sentence) % (2**31))
            label = rng.choice(LABELS)
            confidence = round(rng.uniform(0.45, 0.95), 3)
            results.append(SentenceResult(
                sentence=sentence,
                label=label,
                confidence=confidence,
                line_number=i,
            ))
        return results
