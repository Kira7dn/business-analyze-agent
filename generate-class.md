---
description: Generate Class Orchestrator Workflow
auto_execution_mode: 3
---

Orchestrates the selection of the correct class generation workflow (infrastructure, presentation, domain, application) based on the `layer` field from JSON input.

## Input JSON Schema

```json
[
  {
    "class_name": "Order",
    "layer": "domain/entity",
    "description": "Represents a warehouse order with encapsulated business behavior.",
    "attributes": ["id: int", "items: list", "status: str"],
    "methods": []
  }
]
```

## Rules

- **Routing**:
  - If `layer` starts with `infrastructure/` → use `/generate-infrastructure-class`
  - If `layer` starts with `presentation/` → use `/generate-presentation-class`
  - If `layer` starts with `domain/` → use `/generate-domain-class`
  - If `layer` starts with `application/` → use `/generate-application-class`
- **Consistency**: Pass through all original input fields (`class_name`, `layer`, `description`, `attributes`, `methods`) unchanged.
- **Idempotency**: Only trigger the sub-workflow for classes present in the current JSON input.
- **Isolation**: No logic of code generation inside this file — delegate everything to sub-workflows.

## Steps

### Step 1 – Determine Target Workflow
- Parse JSON `layer` field.
- Select the appropriate sub-workflow according to the rules above.
- Output the workflow name.

**Sample**:
```text
Input: {"class_name": "Order", "layer": "domain/entity"}
Output: "/generate-domain-class"
```

### Step 2 – Execute Target Workflow
- Call the chosen workflow (`/generate-infrastructure-class`, `/generate-presentation-class`, `/generate-domain-class`, `/generate-application-class`).
- Pass the full JSON object for the class as input.

**Sample**:
```json
{
  "class_name": "Order",
  "layer": "domain/entity",
  "description": "Represents a warehouse order...",
  "attributes": ["id: int", "items: list", "status: str"],
  "methods": []
}
```

### Step 3 – Collect Results
- Capture output from the sub-workflow (generated code paths, tests, raw URLs).
- Append results to the updated JSON.
