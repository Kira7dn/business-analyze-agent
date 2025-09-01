---
description: Generate Application Class Workflow
auto_execution_mode: 3
---

Generate application interfaces (ports) and use cases from JSON input with tests. Aligns with Onion/Clean Architecture: Presentation → Application → Domain. Infrastructure implements Application interfaces.

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

Note (Scope): This workflow processes only `application/interface` and `application/use_case`. Other layers are handled by their dedicated workflows.

Cross-ref: See `/.windsurf/workflows/generate-infra-persistence.md` for why persistence ignores the `dependencies` field (repositories receive runtime resources via constructor attributes).

## Variable Naming Convention

- `{base_name}`: Derived from `class_name` by stripping CRUD prefixes/suffixes (e.g., `CreateProductUseCase` → base `product`)
- `{snake_base}`: Snake case of base_name (e.g., `product`)
- `{Base}`: Pascal case of base_name (e.g., `Product`)

## Rules

- **Scope**: Process only items where `layer` is `application/interface` or `application/use_case`. Ignore other layers (handled by their respective workflows).
- **Interfaces (Ports)**: Use `abc.ABC` with `@abstractmethod`, prefix `I<Name>`; place under `application/interfaces/`.
- **Use Cases**: Classes in `application/use_cases/` orchestrate domain via interfaces (ports) only.
- **No Infra in Use Cases**: Do not import SDKs/DB/HTTP. Depend solely on interfaces.
- **Dependencies (from Input JSON)**:
  - For `application/use_case`: use `dependencies` (list of interfaces) to generate `__init__(self, ...)` and store attributes. Each dependency must be an interface (name starts with `I`).
  - For `application/interface`: ignore `dependencies`.
- **Imports**: stdlib (`abc`, `typing`) + `app.domain.entities.*` + `app.application.interfaces.*`.
- **Prohibited**: DB sessions, HTTP calls, file I/O inside use cases.
- **Typing**: Use precise types; `Optional[T]` when JSON says "or None".
- **Determinism**: No randomness/time access inside use cases (pass in values from callers/tests).
- **Idempotency**: Only overwrite files for classes present in the current JSON.
- **Tests**: Mock/fake implementations of interfaces. Use `pytest.mark.unit` and `pytest.mark.parametrize(ids=...)` where applicable. Coverage ≥85%.

## Steps

### Step 1 – Generate Interfaces (Ports)

- **Location**: `backend/app/application/interfaces/{snake_base}.py`
- **Action**: For `layer == "application/interface"`, generate an `I{ClassName}Repository` or `I{ClassName}` interface (as named in JSON) with abstract methods from `methods`.

See Appendix A.1 (Interface Template with Example).

### Step 2 – Generate Use Cases

- **Location**: `backend/app/application/use_cases/{snake_case(class_name)}.py`
- **Action**: For `layer == "application/use_case"`, create a class with `__init__(self, ...)` injecting interface dependencies listed in `dependencies`, and an `execute(...)` method whose signature matches `parameters`.
- **Notes**: Create/transform domain entities inside the use case. Do not use infra directly.

See Appendix A.2 (Use Case Template) and A.5 (Use Case with Multiple Dependencies).

### Step 3 – Generate Tests (Unit)

- **Location**:
  - Use cases: `backend/tests/unit/application/test_use_case_{snake_case}.py`
  - Interfaces: Optionally generate a smoke test to ensure abstract methods cannot instantiate.
- **Action**: For each use case, create tests with fakes/mocks for interfaces. Cover happy-path and edge-cases. Use `pytest.mark.unit`.
- **Notes**:
  - Name test files using the use case class in snake case (e.g., `CreateProductUseCase` → `test_use_case_create_product.py`).
  - Prefer realistic fakes over mocks for simple flows; keep them minimal but interface-compatible.
  - Assert returned types/fields and any side effects on fakes (e.g., increments, stored items).

See Appendix A.3 (Unit Test Template) và A.4 (Fake Repository Pattern).

### Step 4 – Run Tests

```bash
# Unit tests (fast)
.venv/bin/python -m pytest -m "not integration" -v

# Enforce coverage
.venv/bin/python -m pytest --cov=app --cov-report=html --cov-fail-under=85

# Targeted run
.venv/bin/python -m pytest backend/tests/unit -q
```

**Artifacts**: Coverage HTML (`backend/tests/test_output/coverage/`), JUnit XML, logs

### Step 5 – Update JSON with Generated Paths

Add `code_path`, `code_raw_url`, `test_path`, `test_raw_url` for each item.

- Raw URL format: `https://raw.githubusercontent.com/Kira7dn/fastapi_template/main/{path}`

See Appendix A.6 (Updated JSON Sample).

## Notes for DI and Presentation

- Wire concrete implementations in `presentation/api/v1/dependencies/` by returning infrastructure classes that implement these interfaces.
- Endpoints should import use cases and request/response schemas only; inject interfaces via dependencies.

## Appendix

### A.1 Interface Template

```python
# backend/app/application/interfaces/{snake_base}.py
from abc import ABC, abstractmethod
from typing import List, Optional
from app.domain.entities.{snake_base} import {Base}

class I{Base}Repository(ABC):
    """Port for {snake_base} persistence operations."""

    @abstractmethod
    def create(self, {snake_base}: {Base}) -> {Base}: ...

    @abstractmethod
    def get_by_id(self, {snake_base}_id: int) -> Optional[{Base}]: ...

    @abstractmethod
    def get_all(self) -> List[{Base}]: ...
```

### A.2 Use Case Template

```python
# backend/app/application/use_cases/create_{snake_base}_use_case.py
from app.application.interfaces.{snake_base} import I{Base}Repository
from app.domain.entities.{snake_base} import {Base}

class Create{Base}UseCase:
    def __init__(self, repo: I{Base}Repository):
        self.repo = repo

    def execute(self, name: str, description: str | None = None) -> {Base}:
        entity = {Base}(id=0, name=name, description=description)
        return self.repo.create(entity)
```

### A.3 Unit Test Template

```python
# backend/tests/unit/application/test_use_case_create_{snake_base}.py
import pytest
from app.application.use_cases.create_{snake_base}_use_case import Create{Base}UseCase
from app.domain.entities.{snake_base} import {Base}

class InMemory{Base}Repo:
    def __init__(self):
        self.items: list[{Base}] = []

    def create(self, entity: {Base}) -> {Base}:
        entity.id = len(self.items) + 1
        self.items.append(entity)
        return entity

    def get_by_id(self, entity_id: int) -> {Base} | None:
        return next((e for e in self.items if e.id == entity_id), None)

    def get_all(self) -> list[{Base}]:
        return list(self.items)


@pytest.mark.unit
def test_create_{snake_base}_success():
    repo = InMemory{Base}Repo()
    uc = Create{Base}UseCase(repo)
    e = uc.execute(name="Sample", description=None)
    assert isinstance(e, {Base})
    assert e.id == 1
    assert e.name == "Sample"
```

```python
class InMemory{Base}Repo:
    """Minimal, deterministic fake matching the interface used by tests."""
    def __init__(self):
        self.items: list[{Base}] = []

    # Implement only the methods needed by the test scenario.
    # Additional methods can raise NotImplementedError to surface accidental use.
    def create(self, entity: {Base}) -> {Base}:
        entity.id = len(self.items) + 1
        self.items.append(entity)
        return entity
```
