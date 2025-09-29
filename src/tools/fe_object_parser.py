#!/usr/bin/env python3
"""Frontend Object Parser.
Create FE JSON objects from PRD text.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture schema
# ---------------------------------------------------------------------------

LayerName = Literal["domain", "application", "infrastructure", "presentation"]
ComponentType = Literal[
    "domain/entity",
    "domain/service",
    "application/interface",
    "application/use_case",
    "application/store",
    "infrastructure/repository",
    "infrastructure/adapter",
    "presentation/component",
    "presentation/hook",
]

LAYER_ALLOWED_TYPES: Dict[LayerName, List[ComponentType]] = {
    "domain": ["domain/entity", "domain/service"],
    "application": [
        "application/interface",
        "application/use_case",
        "application/store",
    ],
    "infrastructure": ["infrastructure/repository", "infrastructure/adapter"],
    "presentation": [
        "presentation/component",
        "presentation/hook",
    ],
}


class ArchitectureComponent(BaseModel):
    """Define FE object schema following Clean Architecture."""

    name: str = Field(description="PascalCase name for class/hook/component")
    type: ComponentType = Field(
        description=(
            "Architectural classification tag mirroring layer (e.g., 'presentation/hook', 'application/use_case')."
        ),
    )
    layer: LayerName = Field(
        description=(
            "Top-level layer: 'domain' | 'application' | 'infrastructure' | 'presentation'"
        )
    )
    properties: Dict[str, str] = Field(default_factory=dict)
    methods: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# PydanticAI pipelines (agents)
# ---------------------------------------------------------------------------


class FrontendFeature(BaseModel):
    """Frontend feature specification extracted from PRD (framework-agnostic).

    Fields summarize the user goal, actors, key UI steps, and application
    touchpoints to drive FE object generation in later steps.
    """

    name: str = Field(description="Feature PascalCase")
    actor: str = Field(description="User role")
    project_goal: str = Field(description="Business goal of the feature")
    ui_flow: List[str] = Field(description="Important UI steps")
    application_touchpoints: List[str] = Field(default_factory=list)
    noteworthy_constraints: List[str] = Field(default_factory=list)


def prd_to_frontend_feature_agent(
    model_name: str = "openai:o4-mini",
) -> Agent[List[FrontendFeature]]:
    """Create an agent that parses PRD text into a list of FrontendFeature.
    The agent stays framework-agnostic and follows Clean Architecture workflow.
    """
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="low",
        openai_reasoning_summary="detailed",
        temperature=0.2,
    )
    prompt = (
        "You are a Frontend Requirements Analyst.\n"
        "Task: From the given PRD, extract ONLY the essential frontend features.\n\n"
        "RULES:\n"
        "• Include ALL key features explicitly required by the PRD.\n"
        "• Do NOT invent or add extra features beyond what is in the PRD.\n"
        "• Each feature must represent a distinct user or business goal.\n"
        "• Keep the set minimal but sufficient to cover the PRD.\n\n"
        "OUTPUT FORMAT: Strict JSON array. Each element =\n"
        "{\n"
        '  "name": "<short feature name>",\n'
        '  "actor": "<end-user or system actor>",\n'
        '  "project_goal": "<business/user goal this feature fulfills>",\n'
        '  "ui_flow": ["step 1", "step 2", "..."],\n'
        '  "application_touchpoints": ["UseCase", "Repository", "Store", ...],\n'
        '  "noteworthy_constraints": ["UX rule", "security requirement", "performance consideration"]\n'
        "}\n\n"
        "GUIDELINES:\n"
        "• ui_flow = high-level, 3–7 steps max.\n"
        "• application_touchpoints = Clean Architecture entities (usecase, repository, store, service).\n"
        "• noteworthy_constraints = performance, security, UX, or resource constraints.\n"
        "• Generate ONLY the minimal set of features that together cover the PRD.\n\n"
        "EXAMPLE:\n"
        "[{\n"
        '  "name": "User Login",\n'
        '  "actor": "Staff",\n'
        '  "project_goal": "Authenticate securely to access the system",\n'
        '  "ui_flow": ["Open login page", "Enter credentials", "Submit form", "System validates", "Redirect to dashboard"],\n'
        '  "application_touchpoints": ["AuthUseCase", "SessionStore"],\n'
        '  "noteworthy_constraints": ["OAuth2 compliance", "Session timeout after 15 min"]\n'
        "}]\n"
    )

    return Agent(
        model_name,
        output_type=List[FrontendFeature],
        retries=3,
        model_settings=settings,
        system_prompt=prompt,
    )


def feature_to_component_agent(
    model_name: str = "openai:o4-mini",
) -> Agent[List[ArchitectureComponent]]:
    """Create an agent that maps a single FrontendFeature to FE ArchitectureComponents.

    Ensures completeness and linkage across presentation/application/infrastructure.
    """
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="medium",
        openai_reasoning_summary="detailed",
        temperature=0.1,
    )
    prompt = (
        "You are a Frontend Clean Architecture architect."
        "Given a SINGLE frontend feature (project_goal, actor, ui_flow, noteworthy_constraints), design the minimal FEObject set.\n\n"
        "ENUMS (exact):\n"
        "• domain: domain/entity, domain/service\n"
        "• application: application/use_case, application/interface, application/store\n"
        "• infrastructure: infrastructure/adapter, infrastructure/repository\n"
        "• presentation: presentation/component, presentation/hook\n\n"
        "FIELD RULES (framework-agnostic, minimal):\n"
        "• All objects MUST have `metadata.intent`.\n"
        "• presentation/component: metadata.rendering_mode ('server' | 'client'); metadata.invocations [list of {type, name}]; metadata.props {}.\n"
        "• presentation/hook: metadata.rendering_mode must be 'client'; metadata.invocations [list of {type, name}] for stores/use cases/hooks; optional state_fields [list].\n"
        "• application/store: represents a client-side state container. Declare metadata.state_fields [list of keys]; metadata.dependencies [list of use case names]; optional metadata.invocations for helper calls.\n"
        "• application/use_case: metadata.inputs [list], outputs [list], dependencies [list of interfaces/stores/domain services].\n"
        "• application/interface: metadata.contract [list of method names].\n"
        "• infrastructure/repository|adapter: metadata.implements [interface name]; metadata.endpoint [string]; metadata.method [string].\n"
        "• domain/entity|service: metadata.fields [list]; domain/service may declare metadata.dependencies [].\n\n"
        "MANDATORY COMPLETENESS & LINKAGE:\n"
        "• Client components (rendering_mode == 'client') must list hook/store dependencies in metadata.invocations.\n"
        "• Server components (rendering_mode == 'server') may only list use cases or server actions in metadata.invocations; no hooks allowed.\n"
        "• Hooks declare their store/use case invocations and resolve dependencies via factories in `src/presentation/dependency/`.\n"
        "• Use cases depend on interfaces/stores/domain services declared in metadata.dependencies.\n"
        "• Repositories/adapters must implement interfaces referenced by use cases.\n"
        "• Maintain dependency direction: presentation → application → domain.\n"
        "• Provide clear intent for each object in metadata.intent (1 sentence).\n\n"
        "DELIVERABLES:\n"
        "• Minimal set covering the flows (hooks/components + use cases + adapters/stores).\n"
        "• Output ONLY a JSON array. No prose. No null values; use {} or [].\n"
    )

    return Agent(
        model_name,
        output_type=List[ArchitectureComponent],
        retries=3,
        model_settings=settings,
        system_prompt=prompt,
    )


def verify_components_agent(
    model_name: str = "openai:o4-mini",
) -> Agent[List[ArchitectureComponent]]:
    """Create an agent that consolidates and fixes FE objects for consistency.

    Applies naming rules, dependency direction, and fills mandatory metadata.
    """
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="medium",
        openai_reasoning_summary="detailed",
        temperature=0.1,
    )
    prompt = (
        "You are a Frontend Clean Architecture reviewer. Consolidate and fix FEObject definitions to comply with the rules below.\n\n"
        "ENUMS (exact):\n"
        "• domain: domain/entity, domain/service\n"
        "• application: application/use_case, application/interface, application/store\n"
        "• infrastructure: infrastructure/adapter, infrastructure/repository\n"
        "• presentation: presentation/component, presentation/hook\n\n"
        "CHECKLIST:\n"
        "• All objects MUST have metadata.intent (one concise sentence).\n"
        "• Dependency direction: presentation → application → domain.\n"
        "• Naming consistent; remove duplicates/generic names.\n"
        "• No legacy keys (hooks_used, stores_used, use_cases_called, ...).\n"
        "• Only presentation/component may contain metadata.invocations [ {type,name} ]. Hooks must not.\n"
        "• Rendering rules: components default to server; set to client if invocations include hook/store. Hooks must be client.\n"
        "• Infrastructure (repository|adapter) require metadata.implements, metadata.endpoint, metadata.method.\n"
        "• Merge overlapping definitions across features.\n\n"
        "COMPLETENESS FIXES:\n"
        "• Infer and fill minimal metadata (inputs, outputs, dependencies, state_fields, contract, endpoint, method) when obvious.\n"
        "• Enforce naming: interfaces start with 'I' (e.g., IOrderAPI); adapters implement those interfaces.\n"
        "• Ensure linkage is closed: component → hook; hook → use cases; use_case → interfaces/stores; adapter/repository → interfaces.\n"
        "• Keep schemas minimal; preserve intent + linkage.\n\n"
        "RETURN:\n"
        "• JSON array of ArchitectureComponent (layer, type, name, properties, methods, metadata).\n"
    )
    return Agent(
        model_name,
        output_type=List[ArchitectureComponent],
        retries=3,
        model_settings=settings,
        system_prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Parser v2
# ---------------------------------------------------------------------------


class FEObjectParser:
    """Frontend Object Parser v2 orchestrating PRD → features → FE objects → verified set."""

    def __init__(self, model_name: str = "openai:o4-mini") -> None:
        self.model_name = model_name
        self.prd_to_feature = prd_to_frontend_feature_agent(model_name)
        self.feature_to_objects = feature_to_component_agent(model_name)
        self.verify_objects = verify_components_agent(model_name)

    async def process(self, prd_text: str) -> str:
        """Run the full pipeline and return the verified FE objects as JSON string."""
        features: List[FrontendFeature] = (
            await self.prd_to_feature.run(prd_text)
        ).output
        logger.info("Generated %s frontend features", len(features))

        aggregated: List[ArchitectureComponent] = []
        for feature in features:
            logger.info("Generating FE objects for feature: %s", feature.name)
            payload = json.dumps({"feature": feature.model_dump()}, ensure_ascii=False)
            result = (await self.feature_to_objects.run(payload)).output
            aggregated.extend(result)
        logger.info("Generated %s FE objects", aggregated)
        logger.info("Verifying FE objects")
        verify_payload = json.dumps(
            {
                "features": [feature_item.model_dump() for feature_item in features],
                "fe_objects": [object_item.model_dump() for object_item in aggregated],
            },
            ensure_ascii=False,
        )
        verified_objects: List[ArchitectureComponent] = (
            await self.verify_objects.run(verify_payload)
        ).output
        result = [object_item.model_dump() for object_item in verified_objects]
        return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FRONTEND_OUTPUT_DIR = os.path.join("projects", "warehouse", "frontend")
DEFAULT_PRD_PATH = os.path.join("projects", "warehouse", "PRD.md")
DEFAULT_OUTPUT_PATH = os.path.join(FRONTEND_OUTPUT_DIR, "verified_fe_objects.json")


async def main() -> None:
    """CLI entrypoint for running the FE Object Parser."""
    # logging cấu hình
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    logger.setLevel(logging.INFO)

    if not os.path.exists(DEFAULT_PRD_PATH):
        print(f"PRD file not found: {DEFAULT_PRD_PATH}")
        return

    with open(DEFAULT_PRD_PATH, "r", encoding="utf-8") as f:
        prd_text = f.read()

    parser = FEObjectParser()
    print("\n=== FE Object Parser ===")
    fe_objects_json = await parser.process(prd_text)

    os.makedirs(os.path.dirname(DEFAULT_OUTPUT_PATH), exist_ok=True)
    with open(DEFAULT_OUTPUT_PATH, "w", encoding="utf-8") as out:
        out.write(fe_objects_json)

    print(f"Final output saved to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
