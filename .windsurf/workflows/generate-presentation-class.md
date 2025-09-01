---
description: Generate Presentation Class Workflow
auto_execution_mode: 3
---

Generate Presentation layer artifacts: schemas, dependencies, and routers.

## Input JSON Schema

Supported layers: `domain/entity`, `domain/service`, `application/interface`, `application/use_case`, `infrastructure/model`, `infrastructure/repository`, `infrastructure/adapter`, `presentation/schema`, `presentation/dependency`, `presentation/router`.

Note (Scope): This workflow processes only `presentation/schema`, `presentation/dependency`, and `presentation/router`. Other layers are handled by their dedicated workflows.

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

## Variable Naming Convention

- `{base_name}`: Derived from `class_name` by stripping CRUD prefixes/suffixes (e.g., `CreateOrderRequest` → `order`)
- `{snake_base}`: Snake case of base_name (e.g., `order`)
- `{Base}`: Pascal case of base_name (e.g., `Order`)
- `{plural_snake_base}`: Plural snake case (e.g., `orders`)

## Rules

- **Scope**: Process only items where `layer` is `presentation/schema`, `presentation/dependency`, or `presentation/router`. Ignore other layers (handled by their dedicated workflows).
- **Idempotent**: only touch items present in current JSON; same-base DTOs share one schema module.
- **External references**: prefer real imports to upstream contracts/types. If unavailable, keep intended import commented and add minimal local placeholder (see Appendix for Protocol example).
- **Router prefixing**: choose ONE place for prefix - either in `APIRouter(prefix=...)` OR in `include_router(..., prefix=...)`, never both.
- **Pluralization**: use plural snake_case resource names for router prefixes (e.g., `/users`, `/orders`). Irregulars: `person -> people`, `category -> categories`.
- **DI typing**: DI providers must return Application interfaces (Protocols/ABCs), not concrete classes. This keeps Presentation decoupled from Infrastructure.

## Steps

### Step 1 – Schemas

- **Location**: `backend/app/presentation/api/v1/schemas/{snake_case(base_name)}.py`
- **Action**:
  - Derive `base_name`: strip CRUD prefixes (`Create`, `Update`, `Delete`) and suffixes (`Request`, `Response`)
  - Generate Pydantic `BaseModel` classes from JSON `class_name` and `attributes`
  - Map `"name: type"` attributes to model fields
  - Add docstring from `description` if present
  - Group related DTOs (same base name) in one module

### Step 2 – Dependencies

- **Location**: `backend/app/presentation/api/v1/dependencies/{snake_case(base_name)}.py`
- **Action**:
  - Create DI provider functions: `get_{snake_base}_repo()`, `get_{snake_base}_adapter()`, or `get_{snake_base}_client()`
  - Return concrete implementations typed as Application interfaces
  - For repositories: inject `db: Session = Depends(get_db)`
  - Use Protocol placeholder if interface not available (see Appendix A.4)

### Step 3 – Routers

- **Location**: `backend/app/presentation/api/v1/routers/{snake_case(base_name)}.py`
- **Action**:
  - Create `router = APIRouter(prefix="/{plural_snake_base}", tags=["{plural_snake_base}"])`
  - Generate endpoint functions from JSON `methods`
  - Inject dependencies via `Depends(get_{snake_base}_repo)`
  - Call use cases: `{Action}{Base}UseCase(repo).execute(...)` and return response schema
  - Use case naming: `Create/Get/List/Update/Delete{Base}UseCase`
  - Keep handlers thin: validate → use case → return DTO

Validation & Error Handling

- Validate requests via Pydantic schemas. Prefer constrained types and validators in schemas when possible.
- Map exceptions to HTTP:
  - `ValueError`/Pydantic `ValidationError` → 400/422
  - `LookupError` → 404
  - `PermissionError` → 403
  - Unexpected → 500 (avoid leaking details)

API Error Schema (suggested)

```json
{
  "type": "string",
  "title": "string",
  "detail": "string",
  "status": 400,
  "instance": "string",
  "errors": { "field": ["message"] }
}
```

Note: For naming and irregular plurals, see the "Naming & Pluralization Rules" section in `workflow_llm_friendly.md`.

### Step 4 – Register Routers

- **Location**: `backend/app/presentation/main.py`
- **Action**:
  - Import router: `from app.presentation.api.v1.routers.{snake_base} import router as {snake_base}_router`
  - Include in app: `app.include_router({snake_base}_router)`
  - Avoid double prefix (router already has prefix)

### Step 5 – Tests

- **Location**: `backend/tests/e2e/test_{snake_base}_api.py`
- **Action**:
  - Use `httpx.AsyncClient` with FastAPI app
  - Override dependencies: `app.dependency_overrides[get_{snake_base}_repo] = lambda: MockRepo()`
  - Test endpoints with representative payloads
  - Assert status codes (201 for create, 200 for get/update, 204 for delete)
  - Assert response structure and required fields

Tham khảo: Appendix A.3 (Router) và A.4 (API/E2E Test) để xem ví dụ chi tiết.

### Step 6 – Update JSON

- **Action**: Add `code_path` and `code_raw_url` fields to each generated class in JSON
- **Format**: `https://raw.githubusercontent.com/Kira7dn/fastapi_template/main/{path}`

### Step 7 – Run Tests

```bash
.venv/bin/python -m pytest -m e2e -v
```

## Artifacts

- Schemas: `backend/app/presentation/api/v1/schemas/{snake_base}.py`
- Dependencies: `backend/app/presentation/api/v1/dependencies/{snake_base}.py`
- Routers: `backend/app/presentation/api/v1/routers/{snake_base}.py`
- Tests: `backend/tests/e2e/test_{snake_base}_api.py`

## Appendix

### A.1 Schema Template

```python
# backend/app/presentation/api/v1/schemas/{snake_base}.py
from pydantic import BaseModel
from typing import Optional

class Create{Base}Request(BaseModel):
    """Request DTO for creating a {snake_base}."""
    name: str
    description: Optional[str] = None

class {Base}Response(BaseModel):
    """Response DTO representing a {snake_base}."""
    id: int
    name: str
    description: Optional[str] = None
```

### A.2 Dependency

```python
# Repository DI — DB-backed repository
# backend/app/presentation/api/v1/dependencies/{snake_base}.py
from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.db import get_db

from app.application.interfaces.some_repo import ISomeRepo
from app.infrastructure.repositories.some_repo import SomeRepo

def get_some_repo(db: Session = Depends(get_db)) -> ISomeRepo:
    """Repository provider: persistence-focused operations."""
    return SomeRepo(db)
```

```python
# Adapter DI — External HTTP/SDK client
# backend/app/presentation/api/v1/dependencies/{name}.py
from fastapi import Depends

from app.application.interfaces.payment_adapter import IPaymentAdapter
from app.infrastructure.adapters.payment_adapter import PaymentAdapter
from app.core.config import settings  # read from env-backed config

def get_payment_adapter() -> IPaymentAdapter:
    """Adapter/Client provider: integration with external payment service."""
    # Adapt attribute names to your settings schema
    return PaymentAdapter(base_url=settings.payment_base_url, token=settings.payment_token)
```

### A.3 Router

```python
# backend/app/presentation/api/v1/routers/{snake_base}.py
from fastapi import APIRouter, Depends
from app.presentation.api.v1.dependencies.{snake_base} import get_{snake_base}_repo
from app.presentation.api.v1.schemas.{snake_base} import Create{Base}Request, {Base}Response
from app.application.use_cases.create_{snake_base}_use_case import Create{Base}UseCase
from app.application.use_cases.get_{snake_base}_use_case import Get{Base}UseCase

router = APIRouter(prefix="/{plural_snake_base}", tags=["{plural_snake_base}"])

@router.post("", response_model={Base}Response, status_code=201)
def create_{snake_base}(request: Create{Base}Request, repo=Depends(get_{snake_base}_repo)):
    entity = Create{Base}UseCase(repo).execute(
        name=request.name,
        description=request.description
    )
    return {Base}Response.model_validate(entity)

@router.get("/{id}", response_model={Base}Response)
def get_{snake_base}(id: int, repo=Depends(get_{snake_base}_repo)):
    entity = Get{Base}UseCase(repo).execute(id)
    return {Base}Response.model_validate(entity)
```

### A.4 API/E2E Test

```python
# Example uses "Product" as an illustration; adapt names to your {Base}.
# backend/tests/e2e/test_product_api.py
import pytest
from httpx import AsyncClient
from app.presentation.main import app

@pytest.mark.asyncio
@pytest.mark.e2e
async def test_product_create_flow():
    class InMemoryRepo:
        def __init__(self):
            self.items = []
        def create(self, product):
            product.id = len(self.items) + 1
            self.items.append(product)
            return product

    # Override DI for testing
    from app.presentation.api.v1.dependencies.product import get_product_repo
    app.dependency_overrides[get_product_repo] = lambda: InMemoryRepo()

    async with AsyncClient(app=app, base_url="http://test") as ac:
        data = {"name": "Laptop", "category": "electronics", "price_range": "high"}
        res = await ac.post("/products", json=data)
        assert res.status_code == 201
        body = res.json()
        assert body["id"] > 0

    app.dependency_overrides.clear()

Note: Prefer realistic fakes over heavy mocks. See `generate-application-class.md` Appendix for fake repository patterns you can reuse across layers.
```
