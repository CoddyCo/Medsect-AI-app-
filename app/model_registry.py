"""
Model Registry — singleton that owns all model instances.

Adding a new model:
  1. Create a class in app/models/model_N_xxx.py
  2. Import it here
  3. Add an instance to the MODELS list in _build_registry()
  That's it — no other files need touching.
"""
import os
import logging
from typing import Optional

from .models.base import BaseMedModel
from .models.model_0_naive import Model0Naive
from .models.model_1_conv1d import Model1Conv1D
from .models.model_2_minilm import Model2MiniLM
from .models.model_3_char import Model3Char
from .models.model_4_hybrid import Model4Hybrid
from .models.model_5_quadra import Model5Quadra

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Holds all model instances and exposes a lookup interface."""

    def __init__(self):
        self._models: dict[str, BaseMedModel] = {}
        self._build_registry()

    def _build_registry(self):
        import sys
        print("====== STARTING MODEL REGISTRY ======", file=sys.stderr)
        
        print("Loading Model0Naive...", file=sys.stderr)
        m0 = Model0Naive()
        
        print("Loading Model1Conv1D...", file=sys.stderr)
        m1 = Model1Conv1D()
        
        print("Loading Model2MiniLM...", file=sys.stderr)
        m2 = Model2MiniLM()
        
        print("Loading Model3Char...", file=sys.stderr)
        m3 = Model3Char()
        
        print("Loading Model4Hybrid...", file=sys.stderr)
        m4 = Model4Hybrid()
        
        print("Loading Model5Quadra...", file=sys.stderr)
        m5 = Model5Quadra()
        
        print("====== ALL MODELS LOADED ======", file=sys.stderr)

        models = [m0, m1, m2, m3, m4, m5]
        for m in models:
            self._models[m.MODEL_ID] = m
            status = "MOCK  ⚠️ " if m.is_mock else "REAL  ✅"
            logger.info("Registered %s (%s) [%s]", m.MODEL_ID, m.MODEL_NAME, status)

        # ── Startup summary banner ─────────────────────────────────────────
        sep = "─" * 60
        logger.info(sep)
        logger.info("  MedSectAI — Model Registry Startup Summary")
        logger.info(sep)
        real_count = sum(1 for m in models if not m.is_mock)
        mock_count = len(models) - real_count
        for m in models:
            status = "REAL  ✅" if not m.is_mock else "MOCK  ⚠️ "
            logger.info("  [%s]  %-28s  %s", m.MODEL_ID, m.MODEL_NAME, status)
        logger.info(sep)
        logger.info(
            "  %d/%d models loaded from real weights — %d using mock fallback",
            real_count, len(models), mock_count,
        )
        if mock_count > 0:
            logger.warning(
                "  ⚠️  Some models are in MOCK mode. "
                "Check saved_models/ contains the correct .keras / .pkl files."
            )
        logger.info(sep)

    def get(self, model_id: str) -> Optional[BaseMedModel]:
        return self._models.get(model_id)

    def all(self) -> list[BaseMedModel]:
        return list(self._models.values())

    def info(self) -> list[dict]:
        return [m.info() for m in self.all()]


# Module-level singleton — imported by main.py
registry = ModelRegistry()
