---
description: Generate Infrastructure Adapter (External Clients/SDK/HTTP)
auto_execution_mode: 3
---

Generate Infrastructure adapters that implement Application layer ports to external services (HTTP/SDK/queues). Adapters must be DI-first, reliable (timeouts/retries/idempotency), and return DTOs or domain entities (never raw SDK responses).

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

Note (Scope): This workflow processes only `infrastructure/adapter`. Other layers are handled by their dedicated workflows.

## Rules

- **Location**: `backend/app/infrastructure/adapters/{snake_case(class_name)}.py`
- **Naming**: filename = snake_case of `class_name`
- **Config**: always read from `app.core.config.settings` (env-driven), never hardcode secrets
- **Reliability**: enforce default 5s timeout, retries (2 attempts, 300ms backoff), idempotency key where relevant
- **Errors**: catch SDK/HTTP errors and map to typed exceptions (`ExternalBadRequest`, `ExternalRateLimitError`, `ExternalServerError`, `ExternalServiceError`)
- **Return types**: only DTOs or domain entities (no raw SDK objects)
- **Idempotent**: only touch files present in current JSON input
- **No side effects**: Do not create HTTP/SDK clients, connections, or read settings at module import time. All dependencies must be injected or initialized inside the constructor.
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

### Step 1 – Provider Discovery

- **Goal**: Determine transport (HTTP/SDK) and extract endpoint or SDK function hints from official docs.
- **Tools**:

  - `mcp1_resolve-library-id` → get library ID from provider name
  - `mcp1_get-library-docs` → read SDK docs / function
  - `mcp0_brave_web_search` → web search ("<provider> SDK <method> docs", "<provider> REST API <method>")
  - `read_url_content` → parse docs content
  - `grep_search`, `codebase_search` → check if client/provider already exists in repo

- **Process**:

  1. Infer provider from `class_name` / `description` / `method_name`.
  2. Resolve SDK docs using `mcp1_resolve-library-id` + `mcp1_get-library-docs`.
  3. If unclear, use `mcp0_brave_web_search`.
  4. Use `read_url_content` to confirm method/path or SDK function.
  5. Check codebase using `grep_search` or `codebase_search`.

- **Output**:

  - `transport`: `http` | `sdk` | `generic`
  - `suggestion`: `{method, path, required_headers}` or `{package, function_name}`
  - `docs_url`: official documentation link

### Step 2 – Generate Adapter

**Goal**: Implement adapter class based on JSON input, Provider Discovery output, and the Rules.

**Detailed actions:**

1. **File setup**

   - Create file at `backend/app/infrastructure/adapters/{snake_case(class_name)}.py`.
   - Import `settings`, typing modules, and optional SDK/http client.
   - Apply the import strategy defined in Rules (real imports for found, commented for missing).

2. **Class definition**

   - Define class with name = `class_name`.
   - Inherit from Application Port. If missing, follow Rules (local placeholder + commented import).

3. **Constructor (`__init__`)**

   - Accept API key, base_url, and client (http/SDK) as parameters.
   - Use env-driven defaults from `settings` if not provided.
   - Assign injected client to `self.http` or `self.sdk`.
   - Ensure no client is created at module level (DI-first).
   - Support injecting: HTTP client, SDK client, retry policy, logger/tracer.

4. **Method generation**

   - For each method in JSON input:
     - Create function with matching signature using type annotations per Rules.
     - Insert discovered transport details (HTTP method/path or SDK function call).
     - Apply timeout (default 5s), retries (2 attempts, 300ms backoff), and idempotency key for writes.
     - Wrap errors with `_map_error` to raise typed exceptions.
     - Add structured logging (method, URL, status, duration; redact secrets).
     - Parse response and pass to `_to_dto`.

5. **Helper methods**

   - `_request` (HTTP request wrapper, handles retries/timeouts).
   - `_call_sdk` (SDK wrapper, handles retries and error capture).
   - `_map_error` (convert external errors to typed exceptions).
   - `_to_dto` (map response to DTO/domain entity).

6. **Placeholders**
   - Follow Rules for any missing external reference.

### Step 3 – Generate Tests

**Goal**: Guarantee correctness of adapter logic and maintain high coverage.

**Detailed actions:**

1. **Unit tests** (`backend/tests/unit/infrastructure/adapters/test_{snake_case(class_name)}.py`):

   - Verify constructor reads `settings` when no args passed.
   - Verify dependency injection works (inject fake client and capture calls).
   - Verify method signatures match input JSON.
   - Verify `_map_error` correctly maps HTTP status/SDK exceptions.
   - Verify `_to_dto` returns correct shape.

2. **Integration tests** (`backend/tests/integration/infrastructure/adapters/test_{snake_case(class_name)}.py`):

   - Use mock HTTP server (`respx`, `requests_mock`) or SDK sandbox.
   - Assert request payload, headers (e.g., Idempotency-Key).
   - Assert DTO mapping against real/sandbox responses.
   - Mark with `@pytest.mark.integration`.

3. Ensure tests are idempotent and do not leak external state.
4. Maintain coverage ≥ 85%.

### Step 4 – Run Tests

```bash
.venv/bin/python -m pytest -q -m "not integration"
.venv/bin/python -m pytest -q backend/tests/unit/infrastructure/adapters
.venv/bin/python -m pytest -q --cov=backend/app --cov-fail-under=85 -m "not integration"
.venv/bin/python -m pytest -q -m integration
```

**Artifacts**: Coverage HTML (`backend/tests/test_output/coverage/`), JUnit XML, logs

### Step 5 – Push & Update JSON

**Update Input JSON** with generated file locations:

- Add `code_path`, `code_raw_url`, `test_path`, `test_raw_url` to each JSON item
- Raw URL format: `https://raw.githubusercontent.com/Kira7dn/fastapi_template/main/{path}`

## Appendix: Quick Examples

### Adapter

```python
# backend/app/infrastructure/adapters/stripe_client.py
from typing import Dict, Optional
# from app.application.interfaces.payment import IPaymentGateway
from app.core.config import settings
import httpx

class StripeClient:
    def __init__(self, api_key: Optional[str] = None, http=None) -> None:
        self.api_key = api_key or getattr(settings, "STRIPE_SECRET_KEY", None)
        self.base_url = getattr(settings, "STRIPE_BASE_URL", "https://api.stripe.com")
        # DI-first: inject http client, fallback to local httpx.Client
        self.http = http or httpx.Client(timeout=5.0)

    def create_payment_intent(self, amount_cents: int, currency: str, metadata: dict) -> Dict:
        response = self.http.post(
            f"{self.base_url}/v1/payment_intents",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Idempotency-Key": "<generate-uuid>"
            },
            json={"amount": amount_cents, "currency": currency, "metadata": metadata},
        )
        if response.status_code >= 400:
            self._map_error(response)
        return self._to_dto(response.json())

    def _map_error(self, response):
        if response.status_code == 400:
            raise ExternalBadRequest("Bad request to Stripe")
        if response.status_code == 429:
            raise ExternalRateLimitError("Rate limited by Stripe")
        if response.status_code >= 500:
            raise ExternalServerError("Stripe server error")
        raise ExternalServiceError("Unexpected Stripe error")

    def _to_dto(self, data: dict) -> Dict:
        return {
            "id": data.get("id"),
            "status": data.get("status"),
            "amount": data.get("amount"),
            "currency": data.get("currency"),
        }
```

### Helper Methods

```python
# HTTP request wrapper with retries/timeouts
def _request(self, method: str, path: str, **kwargs):
    for attempt in range(3):  # 2 retries + original
        try:
            response = self.http.request(
                method, f"{self.base_url}{path}",
                timeout=5.0, **kwargs
            )
            return response
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            if attempt == 2:  # Last attempt
                raise ExternalServiceError(f"Request failed after retries: {e}")
            time.sleep(0.3 * (2 ** attempt))  # Exponential backoff

# SDK wrapper with error capture
def _call_sdk(self, func_name: str, **kwargs):
    try:
        func = getattr(self.sdk, func_name)
        return func(**kwargs)
    except Exception as e:
        raise ExternalServiceError(f"SDK call {func_name} failed: {e}")
```

### Placeholder

```python
# === AUTO-GENERATED PLACEHOLDERS (DO NOT EDIT MANUALLY) ===
class MissingType:
    """Placeholder for missing type. Replace with actual implementation."""
    pass
# === END AUTO-GENERATED PLACEHOLDERS ===
```
