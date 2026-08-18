# Contributing to MedSectAI

Thank you for your interest in contributing! This document outlines the process for contributing to MedSectAI.

---

## Development Setup

Follow the [Quick Start](README.md#-quick-start--local-development) instructions in the README to get your local environment running.

---

## Project Conventions

### Branch Naming
| Type | Pattern | Example |
|---|---|---|
| Feature | `feat/description` | `feat/add-model-6-bert` |
| Bug fix | `fix/description` | `fix/model-registry-crash` |
| Docs | `docs/description` | `docs/api-reference` |
| Refactor | `refactor/description` | `refactor/s3-fetch-retry-logic` |

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
feat: add BERT-based classifier as Model 6
fix: handle empty abstract input gracefully
docs: update deployment guide for new HF region
refactor: simplify model registry lookup
test: add unit tests for s3_fetch module
chore: update tensorflow to 2.16
```

---

## Adding a New Model

All classifier models live in `app/models/`. To add a new one:

1. **Create a new file** `app/models/model_N_name.py`
2. **Inherit from** `BaseModel` in `app/models/base.py`:

```python
from .base import BaseModel, SentenceResult
from typing import List

class ModelNName(BaseModel):
    MODEL_ID   = "model_N"
    MODEL_NAME = "Your Model Name"

    def _load_weights(self):
        # Load your .keras or .pkl file from saved_models/
        ...

    def predict(self, text: str) -> List[SentenceResult]:
        # Preprocess text, run inference, return results
        ...
```

3. **Register it** in `app/model_registry.py` by adding it to the registry list.
4. **Train it** in `notebooks/` or `scripts/train_models.py` and upload weights to S3.
5. **Update** the frontend `frontend/index.html` with the new model's metadata.

---

## Code Style

- Python: [PEP 8](https://pep8.org/) — enforced by `ruff` (if configured)
- Max line length: **100 characters**
- Type hints are encouraged on all public functions

---

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `master`
3. Make your changes and commit following the conventions above
4. Open a Pull Request against `master`
5. Ensure your PR description explains **what** changed and **why**
6. A maintainer will review and merge

---

## Reporting Bugs

Please open an [issue](issues) and include:
- Python version and OS
- Steps to reproduce
- Expected vs. actual behavior
- Relevant logs or error messages
