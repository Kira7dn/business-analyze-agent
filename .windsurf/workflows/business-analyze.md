---
description: Analyze Idea then return Document with detail Features and Requirement
auto_execution_mode: 3
---

1. Load the input idea/requirements file
2. Call `evaluate_requirements` tool with the initial requirements
3. If questions exist
    + Present questions by form on question.md (same folder) and ask user to fill the answer
    + Make clear idea to clear_idea.md (same folder) base on initial and provided answer
    + Call `evaluate_requirements` tool with content from clear_idea.md
    + Loop over until no questions
4. Call `suggest_features` tool with content from clear_idea.md
5. Update return information from `suggest_features` to clear_idea.md
6. Call `stack_search` tool with content from clear_idea.md
7. Update return information from `stack_search` to clear_idea.md
8. Generate product requirement document PRD.md (same folder) from clear_idea.md with following format:
    + Overview
    + Goals & KPIs
    + Users & Roles
    + In/Out of Scope
    + Feature Suggestions by Role
    + Functional Requirements
    + Non-Functional Requirements
    + Data Model (Minimal)
    + API Endpoints (Illustrative)
    + Architecture Overview
    + Observability & Alerts
    + Risks & Mitigations
    + Acceptance Criteria