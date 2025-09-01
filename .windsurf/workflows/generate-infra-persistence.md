---
description: Generate Infrastructure Persistence (Models + Repositories)
auto_execution_mode: 3
---

Generate SQLAlchemy models and database repositories from JSON class definitions. Automatically handles missing domain entities by creating typed placeholders. Includes Alembic migrations.

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

Note (Scope): This workflow processes only `infrastructure/model` and `infrastructure/repository`. Other layers are handled by their dedicated workflows.

Cross-ref: See `/.windsurf/workflows/generate-application-class.md` for how the `dependencies` field is used to generate constructor injection for use cases.

## Rules

- **Scope**: Process only items where `layer` is `infrastructure/model` or `infrastructure/repository`. Ignore other layers (handled by their respective workflows).
- **File paths**: `snake_case(class_name).py`
- **Idempotent**: only process items in current JSON input
- **External references**:
  - Try real imports:
    - Interfaces/ports: `app.application.interfaces.{snake_case(name)}`
    - Domain entities: `app.domain.entities.{snake_case(name)}`
  - If missing:
    - Keep intended import commented.
    - Generate a local placeholder in the same file (dedupe; remove when real exists):
      - Interfaces/ports: minimal class with method stubs raising `NotImplementedError`; inherit from it.
      - Domain entities: minimal class; use quoted annotations.

## Steps

### Step 1 – Generate Model

- **Location**: `backend/app/infrastructure/models/{snake_case(class_name)}.py`
- **Action**: For each item where `layer == "infrastructure/model"`, generate a SQLAlchemy model:
  - **Naming/imports**: file=`snake_case(class_name).py`; class as-is; `from app.infrastructure.database.base import Base`.
  - **__tablename__**: plural snake_case of class (trim trailing `Model`), e.g., `ProductModel`→`products`, `Order`→`orders`.
  - **Columns (type map)**: `int→Integer`, `str→String`, `float→Float`, `bool→Boolean`, `datetime→DateTime`, `date→Date`, `time→Time`, `list|dict→JSON`, `bytes→LargeBinary`, `Decimal→Numeric`, `UUID→String(36)`.
  - **Nullability**: Optional/"or None" → `nullable=True`; else `False`.
  - **Primary key**: `id: int` → `primary_key=True, autoincrement=True`.
  - **Foreign keys (infer)**: `<entity>_id: int` → `ForeignKey("<entities>.id", ondelete=...)`; target = plural snake_case of `<entity>`; `ondelete`: use `constraints.on_delete` if given, else `RESTRICT` (use `SET NULL` when Optional); index FK columns.
  - **Indexes/uniques**: honor JSON `constraints` if present; else infer — unique on `email|username|slug|code|external_id|uuid`; index `*_id`, `created_at`, `updated_at`, `status`; composite unique on (`tenant_id`, one of the unique-like fields).
  - **Scope**: no relationships or business logic; thin table mapping (validation stays in domain entities).

### Step 2 – Process Repositories

- **Location**: `backend/app/infrastructure/repositories/{snake_case(class_name)}.py`
- **Action**: For each item where `layer == "infrastructure/repository"`, generate a repository class:
  - **Naming/paths**: file = `snake_case(class_name).py`; class = `class_name` as-is.
  - **Imports**: `from typing import Any, Optional, List, Dict` + ORM models as needed.
  - **Interface inheritance**: Inherit from repository interface (follow Rules).
  - **Constructor**: `__init__` with exact `attributes` parameters (preserve order). Store session as `self.db_session`.
  - Note: The `dependencies` field from Input JSON is not used here; repositories receive runtime resources (e.g., DB session) via constructor attributes.
  - **Methods**: exact names/parameters/returns per JSON. Use quoted annotations for missing domain types.
  - **Placeholders**: follow Rules for any missing external reference.
  - **Implementation**: use `self.db_session`, wrap writes in transactions, map Domain↔Model via `_to_model/_to_domain` helpers.
  - **Error handling**: `ValueError` (invalid input), `LookupError` (not found), re-raise DB errors.
  - **Constraints**: no side effects, no business logic, use injected session only.

### Step 3 – Generate Alembic Migrations

**Only if Step 1 generated at least one model:**

```bash
.venv/bin/alembic revision --autogenerate -m "add infrastructure models"
.venv/bin/alembic upgrade head
```

### Step 4 – Generate Tests

- **Location**:
  - Unit tests (default):
    - Models: `backend/tests/unit/infrastructure/models/test_{snake_case(class_name)}.py`
    - Repositories: `backend/tests/unit/infrastructure/repositories/test_{snake_case(class_name)}.py`
  - Integration tests (only if real DB I/O):
    - Models/Repos touching DB: `backend/tests/integration/infrastructure/{models|repositories}/test_{snake_case(class_name)}.py`
    - Add `@pytest.mark.integration`
- **Discovery follows project pytest.ini**:
  - `testpaths = backend/tests`
  - `python_files = test_*.py`, `python_classes = Test*`, `python_functions = test_*`
  - Markers available: `integration`, `slow`, `ai`
- **Models tests** (for each `layer == "infrastructure/model"`):
  - Import the generated model class.
  - Assert `__tablename__` naming, primary key presence, nullable flags, and column types for declared attributes.
  - If FKs inferred: assert presence of FK columns and indexes when applicable.
- **Repositories tests** (for each `layer == "infrastructure/repository"`):
  - If domain entities exist: import real entity types; else use quoted annotations and any generated placeholders.
  - Instantiate repository with a mock or in-memory session double (no real DB I/O).
  - Assert constructor stores session as `self.db_session`.
  - For each method signature from JSON: verify method exists and annotations match.
  - Do not assert DB behavior here; only surface-level behavior (signature, basic return/raise contracts if implemented).

### Step 5 – Run Tests

```bash
# Unit (quick, per markers in pytest.ini)
.venv/bin/python -m pytest -q -m "not integration"

# Unit infra-only
.venv/bin/python -m pytest -q backend/tests/unit/infrastructure

# Unit with coverage + threshold
.venv/bin/python -m pytest -q --cov=backend/app \
  --cov-report=term-missing --cov-report=html:backend/tests/test_output/coverage \
  --cov-fail-under=85 -m "not integration"

# Integration (all)
.venv/bin/python -m pytest -q -m integration

# Integration scoped (infra/models)
.venv/bin/python -m pytest -q backend/tests/integration/infrastructure/models -m integration

# Rerun last failures fast
.venv/bin/python -m pytest -q --last-failed --maxfail=1
```

**Artifacts**: Coverage HTML (`backend/tests/test_output/coverage/`), JUnit XML, logs

### Step 6 – Push & Update JSON

**Update Input JSON** with generated file locations:

- Add `code_path`, `code_raw_url`, `test_path`, `test_raw_url` to each item
- Raw URL format: `https://raw.githubusercontent.com/Kira7dn/fastapi_template/main/{path}`

See Appendix: Updated JSON Sample.

## Appendix: Quick Examples

### Model

```python
# backend/app/infrastructure/models/product.py
from sqlalchemy import Column, Integer, String
from app.infrastructure.database.base import Base

class ProductModel(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
```

### Repo

```python
# backend/app/infrastructure/repositories/sql_order_repository.py
from typing import Any
from app.domain.entities.order import Order

class SqlOrderRepository:
    def __init__(self, db_session: Any) -> None:
        self.db_session = db_session
    def save(self, order: Order) -> Order: ...
    def get_by_id(self, order_id: int) -> Order: ...
```

### Placeholder

```python
# === AUTO-GENERATED PLACEHOLDERS (DO NOT EDIT MANUALLY) ===
class MissingEntity:
    """Placeholder for missing entity. Replace with actual implementation."""
    pass
# === END AUTO-GENERATED PLACEHOLDERS ===
```

### Unit tests

```python
# backend/tests/unit/infrastructure/models/test_{snake_case(class_name)}.py
from app.infrastructure.models.{snake_case(class_name)} import {class_name}
def test_meta():
    assert getattr({class_name}, "__tablename__") == "{plural_snake_case}"
    id_col = getattr({class_name}, "id").property.columns[0]
    assert id_col.primary_key is True
```

```python
# backend/tests/unit/infrastructure/repositories/test_{snake_case(class_name)}.py
from typing import get_type_hints
from app.infrastructure.repositories.{snake_case(class_name)} import {class_name}

class DummySession:
    pass

def test_sig():
    assert hasattr({class_name}(db_session=DummySession()), "db_session")
    get_type_hints({class_name}.save)
```

### Integration tests

```python
# backend/tests/integration/infrastructure/models/test_{snake_case(class_name)}_integration.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.infrastructure.database.base import Base
from app.infrastructure.models.{snake_case(class_name)} import {class_name}

engine = create_engine("sqlite+pysqlite:///:memory:")
SessionLocal = sessionmaker(bind=engine)

@pytest.fixture(scope="module", autouse=True)
def _schema():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture()
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()

@pytest.mark.integration
def test_roundtrip(db_session):
    instance = {class_name}()
    db_session.add(instance)
    db_session.commit()
    assert instance.id
```

```python
# backend/tests/integration/infrastructure/repositories/test_{snake_case(class_name)}_integration.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.infrastructure.database.base import Base
from app.infrastructure.repositories.{snake_case(class_name)} import {class_name} as Repository
from app.domain.entities.{entity_snake} import {entity_class}

engine = create_engine("sqlite+pysqlite:///:memory:")
SessionLocal = sessionmaker(bind=engine)

@pytest.fixture(scope="module", autouse=True)
def _schema():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture()
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()

@pytest.mark.integration
def test_save_get(db_session):
    repo = Repository(db_session=db_session)
    entity = {entity_class}(id=0)
    saved = repo.save(entity)
    assert saved.id
    assert repo.get_by_id(saved.id).id == saved.id
```
