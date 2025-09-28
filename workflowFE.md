# Frontend Clean/Onion Architecture (Next.js + TypeScript)

## Introduction

This document outlines the implementation of **Clean Architecture** (also known as Onion Architecture) for a **Next.js frontend application** with TypeScript. It emphasizes separation of concerns, dependency inversion, and testability.

### Key Principles
- **Dependency Direction**: Presentation → Application → Domain
- **Testability**: Each layer can be tested in isolation
- **Maintainability**: Changes in one layer don't affect others
- **Hybrid Support**: Supports both UI and API presentation in Next.js

## Layer Object Types (FE Object Parser Reference)

- **Domain**
  - `domain/entity`: Pure entities/value objects
  - `domain/service`: Stateless coordinators of complex domain rules
- **Application**
  - `application/interface`: Repository/adapter contracts (ports)
  - `application/use_case`: Use case orchestrators calling domain services/entities
  - `application/store`: Client-side state (Zustand)
- **Infrastructure**
  - `infrastructure/repository`: Implement repository interfaces (DB/cache)
  - `infrastructure/adapter`: External API/SDK adapter implementing interfaces
- **Presentation**
  - `presentation/component`: React components/pages
  - `presentation/hook`: Custom hooks bridging components ↔ application

---

## Architecture Overview

### Layers
1. **Domain Layer**: Business entities and rules
2. **Application Layer**: Use cases and interfaces
3. **Infrastructure Layer**: External concerns (DB, APIs)
4. **Presentation Layer**: UI components and API routes

### Dependency Rule
- Inner layers (Domain) don't depend on outer layers
- Outer layers depend on inner layers via interfaces

---

## Layers Explanation

### Domain Layer
- **Purpose**: Core business logic, entities, validation rules
- **Contents**: Pure TypeScript entities, domain services, value objects
- **Dependencies**: None (except standard library)
- **Testing**: Unit tests for entities and pure functions

### Application Layer
- **Purpose**: Orchestrate business operations
- **Contents**: Use cases, domain facades/services orchestration, server actions, repository interfaces, Zustand stores
- **Dependencies**: Domain layer
- **Testing**: Unit tests with mocked interfaces

### Infrastructure Layer
- **Purpose**: Handle external systems
- **Contents**: Repository implementations, adapters (HTTP/SDK), DB clients, external API clients
- **Dependencies**: Application interfaces
- **Testing**: Integration tests

### Presentation Layer
- **Purpose**: Present data to users and external clients
- **Contents**:
  - **UI Presentation**: React components, pages, hooks
  - **API Presentation**: API routes (JSON responses)
- **Dependencies**: Application layer
- **Testing**: Component tests, API response tests

---

## Workflow (Step-by-Step)

### Step 0: Clarify Requirements
- Define input/output, entities, business rules
- Identify acceptance criteria

### Step 1: Domain Layer
- Create entities (TypeScript classes/interfaces)
- Implement business rules and validation
- Design domain services for complex business orchestration (pure functions/classes without external dependencies)

### Step 2: Application Layer
- Define repository interfaces (ports)
- Implement use cases (application services)
- Wire domain services/entities inside use cases
- Declare application stores (Zustand) and server actions

### Step 3: Infrastructure Layer
- Implement repository interfaces using DB/API
- Add adapters (HTTP/SDK clients) that fulfill application interfaces when direct repository implementation is insufficient
- Keep infrastructure ignorant of presentation concerns

### Step 4: Presentation Layer
- Create UI components/hooks
- Create API routes if needed

### Step 5: Testing
- Write tests for each layer
- Integration and E2E tests

---

## Directory Structure

```
frontend/
├── app/                            # Next.js App Router
│   ├── api/                        # API routes (Presentation: JSON responses)
│   │   ├── users/route.ts          # GET /api/users
│   │   └── cart/[userId]/route.ts  # GET/POST /api/cart/:userId
│   ├── dashboard/                  # UI pages (Presentation)
│   ├── globals.css                 # UI styling (Presentation)
│   ├── layout.tsx                  # UI layout (Presentation)
│   └── page.tsx                    # UI pages (Presentation)
├── public/                         # Static assets (Infrastructure)
├── src/                            # Source code by layers
│   ├── domain/
│   │   └── entities/
│   │       ├── Cart.ts            # Domain entity
│   │       └── Cart.test.ts       # Unit test
│   ├── application/
│   │   ├── interfaces/            # Repository interfaces
│   │   ├── stores/                # ✅ State management stores (Zustand)
│   │   │   └── cartStore.ts       # Global cart state with optimistic updates
│   │   ├── use_cases/             # Use cases
│   │   └── server_actions/        # Server actions
│   ├── infrastructure/
│   │   └── repositories/          # Repository implementations
│   ├── presentation/
│   │   ├── components/            # React components
│   │   ├── dependency/            # Dependency injection factories by domain
│   │   │   ├── auth.ts            # Auth-related factories
│   │   │   └── cart.ts            # Cart-related factories
│   │   └── hooks/                 # Custom hooks
│   └── shared/
│       └── di/                    # Dependency injection
├── tests/
│   ├── integration/               # Integration tests
│   └── e2e/                       # E2E tests
├── jest.config.js                 # Jest configuration
├── playwright.config.ts           # Playwright configuration
└── tsconfig.json                  # TypeScript configuration
```

---

## Examples

### Example 1: Add to Cart (Mutation)
- **Domain**: `Cart` entity with `addItem()` method, `CartItem` type
- **Application**:
  - `AddItemToCart` use case, `ICartRepository` interface
  - `cartStore` Zustand store with optimistic updates and rollback
- **Infrastructure**: `CartRepoHttp` (in-memory storage)
- **Presentation**:
  - UI: `useCart` hook (consumes store), `CartSection` component (controlled form)
  - API: `POST /api/cart/:userId` (future implementation)

### Example 2: User Login (Authentication)
- **Domain**: `User` entity
- **Application**: `AuthenticateUser` use case, `getUsersServerAction`
- **Infrastructure**: `UserRepoHttp` (in-memory users)
- **Presentation**:
  - UI: `useLogin` hook, `LoginComponent`
  - API: `GET /api/users`

---

## Testing Strategy

| Layer          | Test Type       | Tools/Notes |
| -------------- | --------------- | ----------- |
| Domain         | Unit test       | Jest, pure TypeScript |
| Application    | Unit test       | Jest + mocked interfaces |
| Infrastructure | Integration     | Jest + MSW for API mocks |
| Presentation   | Unit/Integration| React Testing Library, Jest |

---

## Tools & Recipes

### Dependency Injection
- Use factory functions for dependency injection đặt tại `src/presentation/dependency/`
- Presentation layer gọi các factory để lấy use case/stores đã được wiring với repository hoặc adapter
- Application layer cung cấp các interface/implementation, tránh để components import trực tiếp repository/infrastructure
- Example: `createAuthenticateUser()` trong `src/presentation/dependency/auth.ts` trả về `AuthenticateUser` đã inject `IUserRepository`

### State Management (Zustand)
- **Global State**: Use Zustand stores in **Application Layer** for cross-component state
- **Local State**: Use React `useState` for component-specific state
- **Store Location**: `src/application/stores/` (not in Presentation)
- **Optimistic Updates**: Update UI immediately, rollback on error
- **Example**:
  ```typescript
  // src/application/stores/cartStore.ts
  export const useCartStore = create<CartStore>()(
    subscribeWithSelector((set, get) => ({
      carts: {},
      addToCartOptimistic: async (userId, formData) => {
        // Optimistic update + API call + rollback logic
      }
    }))
  );
  ```

### MSW (Mock Service Worker)
- Mock API responses in tests
- Example: `http.get('/api/cart/:userId', ({ params }) => HttpResponse.json(cartData))`

### Playwright
- E2E testing
- Example: `page.route('**/api/cart/**', route => route.fulfill({ status: 200, json: cartData }))`

---

## Best Practices

- **Naming Convention**:
  - `.ts` for TypeScript files
  - `.tsx` for React components
  - `.test.ts/.tsx` for unit tests
  - `.integration.test.tsx` for integration tests
  - `.spec.ts` for E2E tests

- **Import Rules**:
  - Use path aliases: `@domain/*`, `@application/*`, etc.
  - Presentation only imports Application
  - No direct API calls in components

- **Error Handling**: Use try-catch in server components/actions
- **Security**: Server actions auto-protect against unauthorized access
- **Performance**: Use server components for data fetching

---

## Conclusion

This architecture provides a solid foundation for scalable, testable Next.js applications. Start with the workflow, implement examples, and adapt as needed.

For questions or contributions, refer to the examples and testing strategies.
