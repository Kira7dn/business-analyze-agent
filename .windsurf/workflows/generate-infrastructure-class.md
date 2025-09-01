---
description: Generate Infrastructure Classes (Router Only)
auto_execution_mode: 3
---

Router workflow that delegates infrastructure generation to specialized sub-workflows. Keeps logic out of this file. Aligns with Onion/Clean Architecture: Infrastructure implements Application ports.

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

Note (Scope): This router handles only `infrastructure/model`, `infrastructure/repository`, and `infrastructure/adapter` items by delegating to sub-workflows:
- `infrastructure/model` or `infrastructure/repository` → `/generate-infra-persistence`
- `infrastructure/adapter` → `/generate-infra-adapter`
Other layers are routed by their dedicated workflows.

## Rules

- **Routing**:
  - If `layer` is `infrastructure/model` or `infrastructure/repository` → use `/generate-infra-persistence`
  - If `layer` is `infrastructure/adapter` → use `/generate-infra-adapter`
- **Pass-through**: Forward the full JSON object unchanged to the sub-workflow.
- **Idempotency**: Only trigger sub-workflows for classes present in the current JSON input.
- **Collection**: Append returned `code_path`, `code_raw_url`, and any `test_*` fields back into the corresponding JSON item.

## Steps

### Step 1 – Determine Target Workflow (per item)

- Parse each JSON object's `layer` field and choose the sub-workflow:
  - `infrastructure/model` or `infrastructure/repository` → `/generate-infra-persistence`
  - `infrastructure/adapter` → `/generate-infra-adapter`

### Step 2 – Execute Target Workflow

- Call the selected sub-workflow with the full JSON object for the class.
- Do not transform input; pass through all fields unchanged.

### Step 3 – Collect Results

- Capture output fields from sub-workflows (e.g., `code_path`, `code_raw_url`, `test_path`, `test_raw_url`).
- Merge these into the corresponding JSON item.
- Raw URL format: `https://raw.githubusercontent.com/Kira7dn/fastapi_template/main/{path}`

### Notes

- This file contains no generation logic. All code scaffolding lives in:
  - `/.windsurf/workflows/generate-infra-persistence.md`
  - `/.windsurf/workflows/generate-infra-adapter.md`
