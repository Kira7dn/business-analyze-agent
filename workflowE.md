# Feature Delivery Workflow (Onion/Clean Architecture)

This document is a repeatable, end-to-end guide for delivering any feature.
It covers: General rules, directory map, step-by-step workflow, and code samples.

## I. General Rules

- Use your virtual environment (venv) for all commands:
  - venv/bin/python -m pytest -q
  - venv/bin/alembic revision --autogenerate -m "<msg>" && venv/bin/alembic upgrade head
- One-way dependency direction: Presentation → Application → Domain; Infrastructure implements Application interfaces.
- Use Cases must not call SDKs directly; they call interfaces in `application/interfaces/`.
- Configuration/secrets live in `backend/app/core/config.py` and env files (.env/.env.docker). Never hardcode keys.
- Thin endpoints: validate → call Use Case → return schema.
- Write tests as soon as you add a Use Case. Prefer mocking Application interfaces.
- External API: Real SDK/API calls live in Infrastructure. Application defines interfaces (ports) + orchestration. Presentation only does DI/routers. Domain stays pure, with no I/O.
- Testing markers & venv: use `-m "not integration"` for fast unit runs; `-m integration` for integration; enable coverage in `pytest.ini`; use `httpx.AsyncClient` for API tests.
- DB sessions: Use per-request session via `app/core/db.py::get_db` with engine/sessionmaker from `backend/app/infrastructure/database/base.py`; keep transactions short and commit/rollback on exit.

## II. Directory Map (locations)

- Domain: `backend/app/domain/`
  - `entities/`, `schemas/`, `services/`
- Application: `backend/app/application/`
  - `interfaces/` (ports: LLM/Media/Vector/Repo/Queue…)
  - `use_cases/` (orchestrate logic)
- Infrastructure: `backend/app/infrastructure/`
  - `database/`
  - `repositories/`
  - `adapters/`,
  - `models/`
- Presentation: `backend/app/presentation/`
  - `api/v1/dependencies/`, `api/v1/routers/`, `main.py`
- Config: `backend/app/core/config.py`
- Alembic: `backend/alembic/` + `backend/alembic.ini`

## III. Delivery Workflow (step-by-step)

1. Clarify requirements: Acceptance Criteria, input/output, latency, security, data/migration.
2. Execution decision: Sync (<~2s) vs Background; whether a Pipeline is needed.
3. Layered design: adjust Domain rules if needed; define Application interfaces; Use Case orchestration.
4. Create interfaces in `application/interfaces/` (LLM/Media/Vector/Repo…). (see: [Step 2 – Application](#step-2-application-layer-orchestrate-use-cases-and-interfaces))
5. Implement Use Cases in `application/use_cases/` (no direct SDK calls). (see: [Step 2 – Application](#step-2-application-layer-orchestrate-use-cases-and-interfaces))
6. Create Infrastructure Repositories/Adapters (can stub first) under `infrastructure/…`. (see: [Step 3 – Infrastructure](#step-3-infrastructure-layer-persistence-external-adapters-pipelines))
7. Wire DI in `presentation/api/v1/dependencies/` and routers in `presentation/api/v1/routers/`, register in `presentation/main.py`. (see: [Step 4 – Presentation](#step-4-presentation-layer-api-endpoints))
8. Configure in `core/config.py` (API keys/model/timeout); add env.
9. Data & Migrations (if any) with Alembic (run inside venv).
10. Testing: Unit (Application) + Integration (Infrastructure) + E2E (Presentation). (see: [Step 5 – Tests](#step-5-tests-unit-integration-e2e))
11. Observability/Resilience: logs/metrics/traces, retries/backoff, idempotency.
12. Security: Pydantic validation, secrets via env, avoid logging PII, RBAC/AuthN/AuthZ.
13. Rollout/Perf: feature flags, caching/batching, timeouts, graceful degradation.

## IV. Detailed Reference

#### Quick TOC

- [Step 0 – Clarify requirements](#step-0-clarify-feature-requirements)
- [Step 1 – Domain](#step-1-domain-layer-core--pure-business-logic)
- [Step 2 – Application](#step-2-application-layer-orchestrate-use-cases-and-interfaces)
- [Step 3 – Infrastructure](#step-3-infrastructure-layer-persistence-external-adapters-pipelines)
- [Step 4 – Presentation](#step-4-presentation-layer-api-endpoints)
- [Step 5 – Tests](#step-5-tests-unit-integration-e2e)

### Step-by-step guide

#### Step 0: Clarify Feature Requirements

- Description: Gather and clarify user requirements. Identify entities, business (including AI) logic, use cases, and initial risks.
- Actions:
  - Ask details: Related entities? Expected input/output? How AI integrates (real-time vs offline)? Constraints (data volume, security)?
  - Risk analysis: For AI features, compute overhead (solution: cache results); validate input to avoid bad prompts/data (Pydantic); DB integration (e.g., store embeddings if needed).
- Example feature:
  - Entities: Product (id, name, category, price_range).
  - AI logic: Recommendation service using cosine similarity over vectors derived from category and price_range.
  - Use cases: Create a new product; Recommend similar products based on user preferences (e.g., list of categories).
  - Risks: Large data may be slow (solution: vector DB like pgvector); Invalid input may crash AI (solution: Pydantic validation).

After clarifying, implement layers sequentially.

#### Step 1: Domain Layer (Core – Pure Business Logic)

- Description: Focus on independent entities and services, with no DB or external deps. Use Pydantic for built-in validation.
- Actions:
  - Define entities with Pydantic BaseModel.
  - Define services containing business/AI logic (import libs like numpy, sklearn as needed).
  - Risks: Complex AI logic is hard to test (solution: keep it pure, train offline if needed); Missing validation (solution: Pydantic validators).

```python
# backend/app/domain/entities/product.py (minimal example)
from pydantic import BaseModel, field_validator

class Product(BaseModel):
    id: int
    name: str
    category: str  # 'electronics' | 'books' | 'clothing'
    price_range: str  # 'low' | 'medium' | 'high'

    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        allowed = ['electronics', 'books', 'clothing']
        if v not in allowed:
            raise ValueError(f"Invalid category: {v}")
        return v
```

#### Step 2: Application Layer (Orchestrate Use Cases and Interfaces)

- Description: Define repository interfaces and use cases to connect domain to infrastructure. Use cases orchestrate logic, call external services via interfaces, and may validate inputs using Pydantic when needed.
- Actions:
  - Define interfaces (ABC) for repositories.
  - Define use cases that inject repos and call services.
  - Risks: Complex use cases create tight coupling (solution: keep orchestration simple and mockable); Unvalidated AI inputs (solution: Pydantic in entities).

```python
# backend/app/application/interfaces/product.py
from abc import ABC, abstractmethod
from app.domain.entities.product import Product

class IProductRepository(ABC):
    @abstractmethod
    def create(self, product: Product) -> Product: ...
    @abstractmethod
    def get_by_id(self, product_id: int) -> Product: ...
    @abstractmethod
    def get_all(self) -> list[Product]: ...

# backend/app/application/use_cases/product.py
from app.application.interfaces.product import IProductRepository
from app.domain.entities.product import Product

class CreateProductUseCase:
    def __init__(self, repo: IProductRepository):
        self.repo = repo
    def execute(self, name: str, category: str, price_range: str) -> Product:
        return self.repo.create(Product(id=0, name=name, category=category, price_range=price_range))
```

#### Step 3: Infrastructure Layer (Persistence, External Adapters, Pipelines)

- Description:
  - Covers persistence (models, repositories, Alembic migrations), concrete external adapters (SDKs/HTTP/queues), and pipeline storage/observability.
- Actions (split by concern):
  - Persistence (DB): Define SQLAlchemy models under `backend/app/infrastructure/models/`; implement repositories mapping domain entities to models; create/run Alembic migrations.
  - External Adapters: Implement clients under `backend/app/infrastructure/adapters/` with retries/backoff, timeouts, idempotency; wire via FastAPI dependencies.
  - Pipeline: Add storage models for `pipeline_runs`, `pipeline_steps`, `pipeline_artifacts` and repositories; compose steps and run via `SimplePipeline`; persist progress when durability is required.

Implementation guide

- Persistence (DB)

  - Place models in `infrastructure/models/` and repository implementations in `infrastructure/repositories/`.
  - Migrations (venv): `venv/bin/alembic revision --autogenerate -m "<msg>" && venv/bin/alembic upgrade head`.
  - Application layer should depend only on interfaces (ports), not SQLAlchemy types.
  - DB sessions: Use per-request session via `app/core/db.py::get_db` with engine/sessionmaker from `backend/app/infrastructure/database/base.py`; keep transactions short and commit/rollback on exit.

  Example (model)

  ```python
  # backend/app/infrastructure/models/product.py (example)
  from sqlalchemy import Column, Integer, String
  from app.infrastructure.database.base import Base

  class ProductModel(Base):
      __tablename__ = "products"
      id = Column(Integer, primary_key=True, index=True)
      name = Column(String, nullable=False)
      category = Column(String, nullable=False)
      price_range = Column(String, nullable=False)
  ```

  Example (repository)

  ```python
  # backend/app/infrastructure/repositories/product.py (simplified)
  from sqlalchemy.orm import Session
  from app.application.interfaces.product import IProductRepository
  from app.domain.entities.product import Product
  from app.infrastructure.models import ProductModel  # example

  class ProductRepository(IProductRepository):
      def __init__(self, db: Session):
          self.db = db
      def create(self, product: Product) -> Product:
          db_product = ProductModel(name=product.name, category=product.category, price_range=product.price_range)
          self.db.add(db_product); self.db.commit(); self.db.refresh(db_product)
          return Product(id=db_product.id, name=db_product.name, category=db_product.category, price_range=db_product.price_range)
  ```

  Note: `ProductModel` and module `app.infrastructure.models` are illustrative. Adjust imports/model names to match your actual codebase.

- External Adapters — Common Guidelines (SDKs/HTTP/Queues)

  - Location: put adapters in `backend/app/infrastructure/adapters/` (one file per service).
  - Contract: implement an Application interface (e.g., `IPaymentGateway`, `ITranscriber`). No domain/application imports in adapters.
  - Config: read from `app/core/config.py` (env-driven). Do not hardcode secrets.
  - Reliability: add retries with backoff, request timeouts, and (optionally) circuit breaker for flaky upstreams.
  - Idempotency: for create-like ops, require an idempotency key or derive one from inputs to avoid duplicates.
  - Errors: catch SDK exceptions; re-raise clean `RuntimeError`/custom errors with actionable messages. Log with contextual metadata.
  - Types: return plain dicts/DTOs, not raw SDK objects. Keep small, serializable payloads.
  - Webhooks: verify signatures in Presentation using the adapter’s helper (e.g., `construct_webhook_event`). Consider persisting raw receipts to `webhook_events`.
  - DI: expose factories in `presentation/api/v1/dependencies/` for injection into endpoints/use cases.
  - Testing: provide lightweight fakes in tests; ensure adapters can be replaced via `app.dependency_overrides`.

  Example (init via settings)

  ```python
  # backend/app/presentation/api/v1/dependencies/payment.py (essentials)
  from app.infrastructure.adapters.payment import StripeClient
  from app.core.config import settings

  def get_payment_gateway():
      return StripeClient(settings.STRIPE_SECRET_KEY)
  ```

- Pipeline

  - When pipelines must be restartable/observable, persist:
    - `pipeline_runs` (run_id, status, started_at, finished_at, error)
    - `pipeline_steps` (run_id, step_name, status, timings)
    - `pipeline_artifacts` (run_id, step_name, type, uri/blob_ref)
  - Provide repositories to read/write these records; wire them via Application ports.

  Example (step skeleton)

  ```python
  # backend/app/infrastructure/pipelines/steps/transcribe_step.py (example)
  from typing import Dict, Any
  from app.application.interfaces.media import ITranscriber

  class TranscribeStep:
      name = "transcribe"
      def __init__(self, transcriber: ITranscriber):
          self.transcriber = transcriber
      def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
          context["text"] = self.transcriber.transcribe(context["audio_path"])
          return context

  # Compose and run (example)
  from app.application.use_cases.run_pipeline import SimplePipeline
  pipeline = SimplePipeline([TranscribeStep(transcriber)])
  result = pipeline.execute({"audio_path": "/tmp/audio.wav"})
  ```

  Note: `transcriber` should be provided via a dependency/factory (e.g., `presentation/api/v1/dependencies/media.py`).

  Example (media DI)

  ```python
  # backend/app/presentation/api/v1/dependencies/media.py (essentials)
  from app.infrastructure.adapters.transcriber import WhisperTranscriber

  def get_transcriber():
      return WhisperTranscriber()
  ```

- Migration Policy

  - Create migrations when adding/modifying persistence models (domain aggregates, pipeline tracking, webhook inbox, etc.).
  - External adapters alone do not require migrations unless they introduce tables (e.g., token cache, webhook inbox).

- Migration commands (venv):
  - `venv/bin/alembic revision --autogenerate -m "add products table"`
  - `venv/bin/alembic upgrade head`

#### Step 4: Presentation Layer (API Endpoints)

- Description: Expose endpoints with FastAPI, use Pydantic for request/response schemas, inject dependencies.
- Actions:
  - Define schemas.
  - Define dependencies and routers.
  - Risks: Slow APIs due to heavy AI (solution: async endpoints); Security holes (solution: Pydantic validation).

```python
# backend/app/presentation/api/v1/schemas/product.py (simplified)
from pydantic import BaseModel

class CreateProductRequest(BaseModel):
    name: str; category: str; price_range: str

class ProductResponse(BaseModel):
    id: int; name: str; category: str; price_range: str

# backend/app/presentation/api/v1/dependencies/product.py
from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.infrastructure.repositories.product import ProductRepository

def get_product_repo(db: Session = Depends(get_db)):
    return ProductRepository(db)

# backend/app/presentation/api/v1/routers/product.py
from fastapi import APIRouter, Depends
from app.application.use_cases.product import CreateProductUseCase
from app.presentation.api.v1.dependencies.product import get_product_repo
from app.presentation.api/v1.schemas.product import CreateProductRequest, ProductResponse

router = APIRouter()

@router.post("/products/", response_model=ProductResponse)
def create_product(request: CreateProductRequest, repo = Depends(get_product_repo)):
    product = CreateProductUseCase(repo).execute(request.name, request.category, request.price_range)
    return ProductResponse.model_validate(product)

# Payments (Stripe)
from app.presentation.api/v1.dependencies.payment import get_payment_gateway
from app.presentation.api/v1.schemas.payment import CreatePaymentIntentRequest, PaymentIntentResponse
from app.application.interfaces.payments import IPaymentGateway

@router.post("/payments/intents", response_model=PaymentIntentResponse)
def create_payment_intent(body: CreatePaymentIntentRequest, gateway: IPaymentGateway = Depends(get_payment_gateway)):
  data = gateway.create_payment_intent(body.amount_cents, body.currency or "", body.metadata)
  return PaymentIntentResponse(**data)
```

```python
# backend/app/presentation/main.py (register routers)
from app.presentation.api.v1.routers import user, ai, payment
app.include_router(ai.router, prefix="/ai", tags=["ai"])
app.include_router(payment.router, prefix="/payments", tags=["payments"])
```

```bash
# Try Payments (Stripe)
curl -X POST 'http://localhost:8006/payments/intents' \
  -H 'Content-Type: application/json' \
  -d '{"amount_cents": 1999, "currency": "usd", "metadata": {"order_id":"ORD-123"}}'

curl 'http://localhost:8006/payments/intents/pi_XXXXXXXXXXXXXXXXXXXX'
```

Environment notes (Stripe): set variables in `.env` or the system before running the API.

- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET` (if using webhooks)
- `STRIPE_DEFAULT_CURRENCY` (default: `usd`)

Example `.env`:

```
STRIPE_SECRET_KEY=sk_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx
STRIPE_DEFAULT_CURRENCY=usd
DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/app
```

Webhook (Stripe):

```python
# backend/app/presentation/api/v1/routers/payment.py (add webhook)
from fastapi import APIRouter, Depends, Request, HTTPException
from app.core.config import settings
from app.application.interfaces.payments import IPaymentGateway
from app.presentation.api/v1.dependencies.payment import get_payment_gateway

router = APIRouter()

@router.post("/webhook")
async def stripe_webhook(request: Request, gateway: IPaymentGateway = Depends(get_payment_gateway)):
    payload = await request.body()  # bytes
    sig_header = request.headers.get("stripe-signature", "")
    try:
        event = gateway.construct_webhook_event(payload, sig_header)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # TODO: handle event types, e.g., 'payment_intent.succeeded'
    # if event["type"] == "payment_intent.succeeded": ...
    return {"received": True}
```

```bash
# Local webhook test (requires stripe CLI):
stripe listen --forward-to localhost:8006/payments/webhook
```

#### Step 5: Tests (Unit, Integration, E2E)

- Description: Follow structure from README: unit (domain/application), integration (infrastructure), e2e (presentation). Use pytest markers, fixtures, and coverage.
- Actions:
  - Unit: Pure logic tests (mock repos).
  - Integration: Test with in-memory DB.
  - E2E: Test API endpoints.
  - Risks: Slow tests (solution: markers like @slow, exclude integration during dev); Low coverage (solution: enforce `--cov-fail-under=85`).
  - Run: `venv/bin/python -m pytest -q` (all); `-m "not integration"` (unit only).
- Concrete examples (add under `tests/unit/`, `tests/integration/`, `tests/e2e/`):

```python
# tests/unit/test_create_product_use_case.py
from app.application.use_cases.product import CreateProductUseCase
from app.domain.entities.product import Product

class InMemoryRepo:
    def __init__(self):
        self.items = []
    def create(self, product: Product) -> Product:
        product.id = len(self.items) + 1
        self.items.append(product)
        return product
    def get_by_id(self, product_id: int) -> Product: ...
    def get_all(self) -> list[Product]: ...

def test_create_product():
    repo = InMemoryRepo()
    uc = CreateProductUseCase(repo)
    p = uc.execute(name="Phone", category="electronics", price_range="high")
    assert p.id == 1 and p.category == "electronics"
```

```python
# tests/integration/test_product_repository.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# TODO: import Base, ProductModel from actual code
from app.infrastructure.repositories.product import ProductRepository
from app.domain.entities.product import Product

@pytest.mark.integration
def test_repo_create(tmp_path):
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    SessionLocal = sessionmaker(bind=engine)
    # Base.metadata.create_all(engine)  # enable when models exist
    db = SessionLocal()
    repo = ProductRepository(db)
    # Flow test (assuming models exist):
    # created = repo.create(Product(id=0, name="Book", category="books", price_range="low"))
    # assert created.id > 0
    assert True  # placeholder so the sample runs
```

```python
# tests/e2e/test_payment_intent_api.py
import pytest
from httpx import AsyncClient
from app.presentation.main import app

@pytest.mark.asyncio
async def test_create_payment_intent(monkeypatch):
    class FakeGateway:
        def create_payment_intent(self, amount_cents, currency, metadata):
            return {"id": "pi_test", "status": "requires_payment_method"}
        def construct_event(self, payload, sig, secret):
            return {"type": "payment_intent.succeeded"}

    # Override dependency if needed
    # app.dependency_overrides[get_payment_gateway] = lambda: FakeGateway()

    async with AsyncClient(app=app, base_url="http://test") as ac:
        res = await ac.post("/payments/intents", json={"amount_cents": 1000, "currency": "usd", "metadata": {}})
        assert res.status_code == 200
        assert res.json()["id"]
```

---

### Supplementary Section (Pipeline, Background Jobs, Testing, Observability, Security)

- Pipeline (optional) – for multi-step workflows (transcribe → embed → index → …):

  - Interfaces in `application/interfaces/pipeline.py`:
    ```python
    # app/application/interfaces/pipeline.py
    from typing import Protocol, Any, Dict
    class IPipelineStep(Protocol):
        name: str
        def run(self, context: Dict[str, Any]) -> Dict[str, Any]: ...
    class IPipeline(Protocol):
        steps: list[IPipelineStep]
        def execute(self, context: Dict[str, Any]) -> Dict[str, Any]: ...
    ```
  - Orchestrator use case (`application/use_cases/run_pipeline.py`) composes steps sequentially.
  - Steps/SDK implementations under `infrastructure/pipelines/steps/…` (e.g., `TranscribeAudioStep`, `GenerateEmbeddingsStep`).
  - DI: create `get_pipeline()` factory in `presentation/api/v1/dependencies/pipeline.py` and an endpoint `/pipeline/run`.

- Background jobs (long tasks):

  - Use a queue (Celery/RQ/Arq) under `infrastructure/jobs/` for heavy video/LLM tasks.
  - Use Case enqueues a job and returns `run_id`; add endpoints `POST /pipeline/run`, `GET /pipeline/status/{run_id}`, `GET /pipeline/result/{run_id}`.

- Testing tips (standard):

  - Structure `tests/`: `unit/`, `integration/`, `e2e/`, and `test_output/` (logs, junit, coverage). Markers: `integration`, `slow`, `ai` (declare in `pytest.ini`).
  - Run in venv: `venv/bin/python -m pytest -q`. Filter with `-m "not integration"`/`-m integration`. Add `--maxfail=3` if needed.
  - API/E2E: use `httpx.AsyncClient` + `ASGITransport`. Override dependencies to mock clients/pipelines.

- Observability & Reliability:

  - Step-level logs with latency; add metrics (Prometheus/OpenTelemetry).
  - Retry/backoff/circuit breaker around SDKs in Infrastructure; ensure idempotency for expensive operations (hash inputs, cache).

- Security:
  - Secrets via env (`core/config.py`), never log PII/keys.
  - Strict validation in Presentation with Pydantic; sanitize file paths/URLs; scan uploads if needed.

#### Wrap-up and Iterations

- After finishing, review everything: run tests with coverage, check risks (e.g., AI performance benchmarks), and deploy (e.g., uvicorn).
- Iteration: If the feature needs refinement, return to Step 0 with new feedback.
- Project planning: stage work (e.g., Week 1: Domain + Application; Week 2: Infrastructure + Presentation; Week 3: Tests + Optimization). What do you think about this example? Do you have a specific feature we should apply this flow to?
