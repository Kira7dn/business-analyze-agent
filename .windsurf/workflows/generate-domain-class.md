---
description: Generate Domain Class Workflow
auto_execution_mode: 3
---

Generate domain classes and services from JSON input with tests and GitHub integration.

## Input JSON Schema

```json
[
  {
    "class_name": "YourClassName",
    "layer": "domain/entity | domain/service | application/interface | application/use_case | infrastructure/model | infrastructure/repository | infrastructure/adapter | presentation/schema | presentation/dependency | presentation/router",
    "description": "Short purpose of this class",
    "attributes": ["field_name: type"],
    "methods": [
      {
        "method_name": "method_name",
        "description": "Optional short method description",
        "parameters": ["arg1: type", "arg2: type"],
        "return_type": "ReturnType"
      }
    ],
    "dependencies": ["IExamplePort"]
  }
]
```

Note (Scope): This workflow processes only `domain/entity` and `domain/service`. Other layers are handled by their dedicated workflows.

## Rules

- **Scope**: Process only items where `layer` is `domain/entity` or `domain/service`. Ignore other layers (they are handled by their own workflows).
- **Entities**: Pydantic `BaseModel` with validators, `model_config = {"extra": "forbid", "validate_assignment": True}`
- **Services**: Pure functions/classes, no side effects
- **Dependencies field**: Present in the canonical schema but not used for domain generation; ignore or validate only for awareness (must not reference infra).
- **Prohibited**: DB, HTTP calls, file I/O
- **Imports**: stdlib + pydantic only
- **Tests**: ≥85% coverage, deterministic, use `pytest.mark.parametrize` with `ids=`
- **Typing**: primitives `int|str|float|bool|datetime|list|dict`; "or None" → `Optional[...]`
- **Idempotency**: only overwrite files for classes in current JSON

## Steps

Process only JSON items with `layer` equal to `domain/entity` or `domain/service`. Ignore other layers.

### Step 1 – Generate Entities

- **Location**: `backend/app/domain/entities/{snake_case(class_name)}.py`
- **Action**: Parse JSON for `layer == "domain/entity"`, create Pydantic BaseModel with:
  - All fields from `attributes`
  - Validators for non-empty strings/lists, non-negative numbers, enum constraints
  - Methods from `methods` array with exact signatures

Tham khảo: Appendix A.1 (Entity Sample).

### Step 2 – Generate Services

- **Location**: `backend/app/domain/services/{snake_case(class_name)}.py`
- **Action**: For each item where `layer == "domain/service"`, generate one pure service class:
  - **Naming/paths**: file=`snake_case(class_name).py`; class=`class_name` as-is (do not add/remove "Service").
  - **Imports/prohibited**: stdlib + `typing` only; import needed entities from `app.domain.entities.{snake_case(entity_name)}`; no third-party, env/framework/loggers/caches/print/external services.
  - **Pure/stateless**: no I/O, globals, randomness, or clocks. Prefer `@staticmethod`; use `@classmethod` only for class-level helpers.
  - **Methods**: implement declared methods exactly; only immutable defaults; precise typing (entities, `Optional[...]`, collections); concise Google-style docstrings (Args/Returns/Raises).
  - **Validation/errors**: validate inputs; raise `ValueError` (bad data) or `PermissionError` (authorization) with specific messages.
  - **Helpers/module**: private static helpers prefixed `_`; public methods first; module = imports then class; no import-time side effects.

Tham khảo: Appendix A.2 (Service Sample).

### Step 3 – Generate Tests

- **Location**: `backend/tests/unit/test_entities_{snake_case}.py`, `test_services_{snake_case}.py`
- **Action**: Create comprehensive test coverage:
  - **Entities**: Valid/invalid cases for each field, method happy-path + edge-cases
  - **Services**: Happy-path + edge-cases for each method
  - Use `pytest.mark.parametrize` with `ids=` for readable case names
  - Deterministic (pass timestamps from tests, no clocks/RNG in code under test)

Tham khảo: Appendix A.3 (Unit Tests Sample).

### Step 4 – Run Tests

```bash
# Standard unit run
.venv/bin/python -m pytest backend/tests/unit

# With coverage enforcement
.venv/bin/python -m pytest backend/tests/unit --cov-fail-under=85

# Exclude non-unit markers
.venv/bin/python -m pytest backend/tests/unit -m "not slow and not integration and not ai"
```

**Artifacts**: Coverage HTML (`backend/tests/test_output/coverage/`), JUnit XML, logs

### Step 5 – Push & Update JSON

**Update Input JSON** with generated file locations:

- Add `code_path`, `code_raw_url`, `test_path`, `test_raw_url` to each item
- Raw URL format: `https://raw.githubusercontent.com/Kira7dn/fastapi_template/main/{path}`

Tham khảo: Appendix A.4 (Updated JSON Sample).

## Appendix

### A.1 Entity Sample

```python
from pydantic import BaseModel, field_validator

class Product(BaseModel):
    model_config = {"extra": "forbid", "validate_assignment": True}

    id: int
    name: str
    category: str

    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        allowed = ['electronics', 'books', 'clothing']
        if v not in allowed:
            raise ValueError(f"Invalid category: {v}")
        return v
```

### A.2 Service Sample

```python
from typing import List
from app.domain.entities.product import Product

class ProductRecommendationService:
    @staticmethod
    def calculate_similarity(product1: Product, product2: Product) -> float:
        category_match = 1.0 if product1.category == product2.category else 0.0
        return category_match

    @classmethod
    def recommend_similar(cls, target: Product, candidates: List[Product], limit: int = 5) -> List[Product]:
        scored = [(p, cls.calculate_similarity(target, p)) for p in candidates if p.id != target.id]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, score in scored[:limit]]
```

### A.3 Unit Tests Sample

```python
import pytest
from pydantic import ValidationError
from app.domain.entities.product import Product

class TestProductEntity:
    def test_create_valid_product(self):
        p = Product(id=1, name="Book A", category="books")
        assert p.id == 1
        assert p.category == "books"

    @pytest.mark.parametrize("bad_category", ["toys", "food", ""], ids=["invalid_toys", "invalid_food", "empty"])
    def test_validate_category_rejects_invalid(self, bad_category):
        with pytest.raises((ValueError, ValidationError)):
            Product(id=1, name="X", category=bad_category)
```

### A.4 Updated JSON Sample

```json
[
  {
    "class_name": "Product",
    "...": "...",
    "code_path": "backend/app/domain/entities/product.py",
    "code_raw_url": "https://raw.githubusercontent.com/Kira7dn/fastapi_template/main/backend/app/domain/entities/product.py",
    "test_path": "backend/tests/unit/test_entities_product.py",
    "test_raw_url": "https://raw.githubusercontent.com/Kira7dn/fastapi_template/main/backend/tests/unit/test_entities_product.py"
  }
]
```
