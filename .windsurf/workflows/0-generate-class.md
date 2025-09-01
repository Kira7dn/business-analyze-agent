---
description: Orchestrates and selects the correct class generation workflow based on the class layer from verified_classes.json.
auto_execution_mode: 3
---

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
