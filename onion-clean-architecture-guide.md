
# Onion/Clean Architecture – Practical Guide (Use Case with single `execute()`)

This guide shows how to design and implement a **standard Use Case** that follows Onion/Clean Architecture with **one public entrypoint: `execute()`**.  
It includes responsibilities per layer, naming conventions, **class definition schema**, **DataFlow mapping rules**, folder structure, and a full **PlaceOrder** example (entities, interfaces, repository/adapter implementations, ORM model, schemas), plus wiring, testing tips, and common pitfalls.

---

## Table of Contents
1. [Core Principles](#core-principles)  
2. [Layer Responsibilities](#layer-responsibilities)  
3. [Standard Artifacts & ClassDefinition Schema](#standard-artifacts--classdefinition-schema)  
4. [Naming Conventions](#naming-conventions)  
5. [DataFlow → Use Case Mapping](#dataflow--use-case-mapping)  
6. [Recommended Folder Structure](#recommended-folder-structure)  
7. [End-to-End Example: PlaceOrder](#end-to-end-example-placeorder)  
   - [Domain](#domain)  
   - [Application](#application)  
   - [Infrastructure](#infrastructure)  
   - [Presentation](#presentation)  
   - [Mermaid Sequence Diagram](#mermaid-sequence-diagram)  
   - [Use Case Pseudo-code](#use-case-pseudo-code)  
8. [Wiring / Composition](#wiring--composition)  
9. [Testing Strategy](#testing-strategy)  
10. [Checklist & Pitfalls](#checklist--pitfalls)

---

## Core Principles

- **Separation of Concerns**: Domain logic is pure and isolated from technical details.  
- **Dependency Rule**: Inner layers must not depend on outer layers. Outer layers depend on inner via interfaces (ports).  
- **Use Case as Orchestrator**: A Use Case represents **exactly one DataFlow**, coordinating domain entities/services with repositories and adapters.  
- **Single entrypoint**: Each Use Case exposes **one public method** `execute()`.  
- **Interfaces define WHAT; Implementations define HOW**.  
- **DI (Dependency Injection)**: Inject interfaces into the Use Case; bind implementations at the boundary (composition root).

---

## Layer Responsibilities

| Layer | Responsibilities | Examples (Artifacts) |
|---|---|---|
| **Domain** | Business concepts and rules, intrinsic behaviors and validations. **No I/O**. | Entities (`Order`, `OrderItem`), Domain Services (pricing rules, discounts) |
| **Application** | Orchestrates workflows. Depends on interfaces (repositories/adapters). **No SDK/DB calls directly**. | `PlaceOrderUseCase`, `IOrderRepository`, `IPaymentAdapter`, `IEmailAdapter` |
| **Infrastructure** | Implements application interfaces with technical details. | `SQLOrderRepository`, `StripePaymentAdapter`, `SMTPEmailAdapter`, ORM models |
| **Presentation** | Input/Output schemas, controllers/routers (not in this guide). | `PlaceOrderRequest`, `PlaceOrderResponse` |

> In this guide, Presentation contains only **schemas**. Controllers/routers are out of scope.

---

## Standard Artifacts & ClassDefinition Schema

Every class should be emitted in a structured form to enable code generation.

### Allowed layer tags
`'domain/entity'`, `'domain/service'`, `'application/use_case'`, `'application/repository_interface'`, `'application/adapter_interface'`, `'infrastructure/repository'`, `'infrastructure/adapter'`, `'infrastructure/model'`, `'presentation/schema'`

### ClassDefinition (canonical fields)

```yaml
class_name: PascalCase
layer: one of the allowed layer tags
description: 1–3 sentences focusing on *why* the class exists and what it owns
attributes:
  - name: type             # list of "name: type" strings
methods:
  - method_name: snake_case
    description: what this method achieves
    parameters:
      - name: type         # list of "name: type" strings
    return_type: string
# Optional — declare interface(s) this class implements or base classes it extends
inheritance:
  - InterfaceOrBaseName
```

### MethodSpec rule
- **Use Case** MUST expose exactly one public method named **`execute`**.  
- For other classes, define only what is necessary to support the Use Case.  
- Add meaningful **descriptions**; keep signatures explicit.

---

## Naming Conventions

- `class_name`: **PascalCase** (e.g., `PlaceOrderUseCase`, `IOrderRepository`, `SQLOrderRepository`).  
- Attributes & method names: **snake_case** (e.g., `order_repo`, `send_order_confirmation`).  
- Repositories: `I<Entity>Repository`, implementations `<Tech><Entity>Repository`.  
- Adapters: `I<Capability>Adapter`, implementations `<Tech><Capability>Adapter`.  
- Optional ORM models under `infrastructure/model` (no behavior, mapping only).

---

## DataFlow → Use Case Mapping

A Use Case maps **1:1** to a DataFlow:

- **`execute()` parameters** = DataFlow **input**.  
- **`execute()` return_type** = DataFlow **output**.  
- **DataFlow steps** = **orchestration inside `execute()`** (no extra public methods).  
- Dependencies are the **interfaces** required to perform steps (repositories/adapters).

---

## Recommended Folder Structure

Traditional **Repository + Adapter** style (DDD-friendly):

```
src/
  domain/
    entity/
    service/
  application/
    use_case/
    repository_interface/
    adapter_interface/
  infrastructure/
    repository/
    adapter/
    model/           # optional ORM models
  presentation/
    schema/
```

> Tip: For larger systems, subfolder by bounded context (e.g., `ordering/`, `billing/`).

---

## End-to-End Example: PlaceOrder

### Domain

#### `Order`
```yaml
class_name: Order
layer: domain/entity
description: Represents a customer's order and owns intrinsic validation and total calculation.
attributes:
  - id: int
  - customer_id: int
  - items: list[OrderItem]
  - total_price: float
  - status: str           # e.g., 'PENDING', 'PAID'
methods:
  - method_name: calculate_total
    description: Sum item subtotals and assign to total_price.
    parameters: []
    return_type: float
  - method_name: validate_items
    description: Validate quantities and positive prices; raise on invalid data.
    parameters: []
    return_type: None
```

#### `OrderItem`
```yaml
class_name: OrderItem
layer: domain/entity
description: Represents a line item in an order, owning quantity and price.
attributes:
  - product_id: int
  - quantity: int
  - price: float
methods:
  - method_name: subtotal
    description: Return quantity * price.
    parameters: []
    return_type: float
```

---

### Application

#### Use Case: `PlaceOrderUseCase`
```yaml
class_name: PlaceOrderUseCase
layer: application/use_case
description: Places an order by validating domain data, persisting it, charging payment, and notifying the customer.
attributes:
  - order_repo: IOrderRepository
  - payment_adapter: IPaymentAdapter
  - email_adapter: IEmailAdapter
methods:
  - method_name: execute
    description: Orchestrate the PlaceOrder DataFlow end-to-end.
    parameters:
      - customer_id: int
      - items: list[OrderItem]
    return_type: Order
```

#### Repository Interface: `IOrderRepository`
```yaml
class_name: IOrderRepository
layer: application/repository_interface
description: Abstracts persistence operations for orders so application code does not depend on DB details.
attributes: []
methods:
  - method_name: save
    description: Persist an order and return the stored entity (with assigned id).
    parameters:
      - order: Order
    return_type: Order
  - method_name: get_by_id
    description: Retrieve an order by its id.
    parameters:
      - order_id: int
    return_type: Order
```

#### Adapter Interface: `IPaymentAdapter`
```yaml
class_name: IPaymentAdapter
layer: application/adapter_interface
description: Abstracts payment processing so the application can charge customers without binding to a specific provider.
attributes: []
methods:
  - method_name: charge
    description: Charge the customer for the given amount and return success/failure.
    parameters:
      - customer_id: int
      - amount: float
    return_type: bool
```

#### Adapter Interface: `IEmailAdapter`
```yaml
class_name: IEmailAdapter
layer: application/adapter_interface
description: Abstracts email sending capability for order notifications.
attributes: []
methods:
  - method_name: send_order_confirmation
    description: Send an order confirmation email.
    parameters:
      - customer_id: int
      - order_id: int
    return_type: None
```

---

### Infrastructure

#### Repository Implementation: `SQLOrderRepository`
```yaml
class_name: SQLOrderRepository
layer: infrastructure/repository
description: Implements IOrderRepository using SQLAlchemy with PostgreSQL.
attributes:
  - db: Session
methods:
  - method_name: save
    description: Map Order to ORM model, upsert it via SQLAlchemy, return the rehydrated domain entity.
    parameters:
      - order: Order
    return_type: Order
  - method_name: get_by_id
    description: Load ORM row by id, map to domain entity, return.
    parameters:
      - order_id: int
    return_type: Order
inheritance:
  - IOrderRepository
```

#### Adapter Implementation: `StripePaymentAdapter`
```yaml
class_name: StripePaymentAdapter
layer: infrastructure/adapter
description: Implements IPaymentAdapter using Stripe SDK.
attributes:
  - client: StripeClient
methods:
  - method_name: charge
    description: Call Stripe to create a charge; return True on success, False otherwise.
    parameters:
      - customer_id: int
      - amount: float
    return_type: bool
inheritance:
  - IPaymentAdapter
```

#### Adapter Implementation: `SMTPEmailAdapter`
```yaml
class_name: SMTPEmailAdapter
layer: infrastructure/adapter
description: Implements IEmailAdapter over SMTP.
attributes:
  - smtp_client: Any
methods:
  - method_name: send_order_confirmation
    description: Compose and send a confirmation email using SMTP client.
    parameters:
      - customer_id: int
      - order_id: int
    return_type: None
inheritance:
  - IEmailAdapter
```

#### ORM Model (Optional): `OrderModel`
```yaml
class_name: OrderModel
layer: infrastructure/model
description: SQLAlchemy ORM model mapping to the 'orders' table.
attributes:
  - id: int
  - customer_id: int
  - total_price: float
  - status: str
methods: []
```

---

### Presentation

#### Schema: `PlaceOrderRequest`
```yaml
class_name: PlaceOrderRequest
layer: presentation/schema
description: Input schema for placing an order.
attributes:
  - customer_id: int
  - items: list[OrderItem]
methods: []
```

#### Schema: `PlaceOrderResponse`
```yaml
class_name: PlaceOrderResponse
layer: presentation/schema
description: Output schema for a successfully placed order.
attributes:
  - order_id: int
  - status: str
  - total_price: float
methods: []
```

---

### Mermaid Sequence Diagram

```mermaid
sequenceDiagram
    participant API as Presentation (API)
    participant UC as PlaceOrderUseCase
    participant DOM as Domain (Order)
    participant REPO as IOrderRepository
    participant PAY as IPaymentAdapter
    participant EMAIL as IEmailAdapter

    API->>UC: execute(customer_id, items)
    UC->>DOM: new Order(customer_id, items)
    UC->>DOM: validate_items(); total = calculate_total()
    UC->>REPO: save(order)
    REPO-->>UC: saved_order (id assigned)
    UC->>PAY: charge(customer_id, total)
    PAY-->>UC: success=true
    UC->>EMAIL: send_order_confirmation(customer_id, saved_order.id)
    EMAIL-->>UC: ack
    UC-->>API: saved_order
```

---

### Use Case Pseudo-code

```python
class PlaceOrderUseCase:
    def __init__(self, order_repo, payment_adapter, email_adapter):
        self.order_repo = order_repo
        self.payment_adapter = payment_adapter
        self.email_adapter = email_adapter

    def execute(self, customer_id, items):
        # 1) Create domain entity
        order = Order(customer_id=customer_id, items=items)
        order.validate_items()
        total = order.calculate_total()

        # 2) Persist
        saved_order = self.order_repo.save(order)

        # 3) Payment
        ok = self.payment_adapter.charge(customer_id=customer_id, amount=total)
        if not ok:
            raise Exception("Payment failed")

        # 4) Notify
        self.email_adapter.send_order_confirmation(customer_id=customer_id, order_id=saved_order.id)

        # 5) Return result
        return saved_order
```

---

## Wiring / Composition

**Example (FastAPI + manual DI):**

```python
# composition_root.py
order_repo = SQLOrderRepository(db=session)
payment_adapter = StripePaymentAdapter(client=stripe_client)
email_adapter = SMTPEmailAdapter(smtp_client=smtp)
place_order_uc = PlaceOrderUseCase(order_repo, payment_adapter, email_adapter)

# presentation/api.py
@app.post("/orders")
def place_order(req: PlaceOrderRequest) -> PlaceOrderResponse:
    order = place_order_uc.execute(req.customer_id, req.items)
    return PlaceOrderResponse(order_id=order.id, status=order.status, total_price=order.total_price)
```

> In real projects, prefer a DI container to wire dependencies per environment (dev/test/prod).

---

## Testing Strategy

- **Domain**: pure unit tests (`Order.validate_items`, `calculate_total`).  
- **Use Case**: unit tests with **mocks/fakes** for `IOrderRepository`, `IPaymentAdapter`, `IEmailAdapter`.  
  - Happy path: save ok, charge ok, email ok.  
  - Failure path: payment fails → ensure proper error handling/compensation.  
- **Infrastructure**: adapter-level tests (integration tests) per technology (DB, Stripe sandbox, SMTP dev server).  
- **Contract tests**: verify implementations match interface semantics.

---

## Checklist & Pitfalls

**Checklist**
- [ ] Each Use Case = exactly one DataFlow; single `execute()`.
- [ ] Application depends only on interfaces, never on tech classes or SDKs.
- [ ] Domain has **no I/O**, only intrinsic behaviors/validations.
- [ ] Repositories/Adapters live in Infrastructure and **implement** their Application interfaces.
- [ ] Optional ORM models are **mapping-only** (no business logic).
- [ ] Presentation exposes *schemas only* (for this spec).

**Common Pitfalls**
- Mixing persistence logic into Domain or Use Case.  
- Letting Use Case call SDKs directly (Stripe, SMTP, etc.).  
- Entities calling `save()` on themselves (Active Record style) — avoid.  
- Over-designing repositories (expose only operations required by the Use Case).  
- Merging multiple DataFlows into a single Use Case.

---

**End of Guide.**
