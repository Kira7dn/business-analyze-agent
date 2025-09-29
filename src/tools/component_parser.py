#!/usr/bin/env python3
"""Component Parser.
Create JSON components from PRD text.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

logger = logging.getLogger(__name__)
load_dotenv()

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
    "infrastructure/models",
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
    "infrastructure": [
        "infrastructure/repository",
        "infrastructure/adapter",
        "infrastructure/models",
    ],
    "presentation": [
        "presentation/component",
        "presentation/hook",
    ],
}


class Component(BaseModel):
    """Define component schema following Clean Architecture."""

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
    feature: str = Field(description="PascalCase name for feature")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# PydanticAI pipelines (agents)
# ---------------------------------------------------------------------------


class Feature(BaseModel):
    """Feature extracted from PRD (framework-agnostic).

    Fields summarize the user goal, actors, key UI steps, and application
    touchpoints to drive FE object generation in later steps.
    """

    name: str = Field(description="Feature PascalCase")
    actor: str = Field(description="User role")
    project_goal: str = Field(description="Business goal of the feature")
    ui_flow: List[str] = Field(description="Important UI steps")


def prd_to_feature_agent(
    model_name: str = "openai:gpt-5-mini",
):
    """Create an agent that parses PRD text into a list of Feature.
    The agent stays framework-agnostic and follows Clean Architecture workflow.
    """
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="low",
        openai_reasoning_summary="detailed",
        temperature=0.2,
    )
    prompt = (
        "You are a Requirements Analyst.\n"
        "Task: From the given PRD, extract ONLY the essential **Functional Features**.\n\n"
        "RULES:\n"
        "• Include ALL key functional features explicitly required by the PRD.\n"
        "• Do NOT include Non-Functional Requirements (e.g., performance, reliability, security, scalability, UI style, maintainability).\n"
        "• Do NOT invent or add extra features beyond what is in the PRD.\n"
        "• Each feature must represent a distinct user or business goal (functional behavior).\n"
        "• Keep the set minimal but sufficient to cover the PRD’s functional scope.\n\n"
        "GUIDELINES:\n"
        "• ui_flow = high-level, 3–7 steps max.\n"
        "• Generate ONLY the minimal set of functional features that together cover the PRD.\n\n"
        "OUTPUT FORMAT: Strict JSON array.\n"
        "EXAMPLE:\n"
        "[{\n"
        '  "name": "User Login",\n'
        '  "actor": "Staff",\n'
        '  "project_goal": "User can login to access the system",\n'
        '  "ui_flow": ["Open login page", "Enter credentials", "Submit form", "System validates", "Redirect to dashboard"]\n'
        "}]\n"
    )

    return Agent(
        model_name,
        output_type=List[Feature],
        retries=3,
        model_settings=settings,
        system_prompt=prompt,
    )


def feature_to_component_agent(
    model_name: str = "openai:gpt-5-mini",
):
    """Create an agent that maps a single Feature to Components.

    Ensures completeness and linkage across presentation/application/infrastructure.
    """
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="medium",
        openai_reasoning_summary="detailed",
        temperature=0.1,
    )
    prompt = (
        "You are a Clean Architecture architect.\n"
        "Given a SINGLE feature (project_goal, actor, ui_flow), design the minimal Component set.\n\n"
        "COMPONENT TYPES (exact):\n"
        "• domain [entity, service]: Show business concept and business process respectively.\n"
        "• application [use_case, interface, store]: Show business process, business contract and client-side state container respectively.\n"
        "• infrastructure [adapter, repository, models]: Show external system and database interaction respectively.\n"
        "• presentation [component, hook]: Show UI component and hook respectively.\n\n"
        "FIELD RULES (framework-agnostic, minimal):\n"
        "• All objects MUST have `metadata.intent`.\n"
        "• Naming: interfaces start with 'I' (e.g., IOrderAPI); adapters implement those interfaces.\n"
        "• Field `feature` MUST equal the provided feature.name for every component.\n"
        "• presentation/component: metadata.rendering_mode ('server' | 'client'); metadata.invocations [list of use cases(server) or hooks(client)]; metadata.props [list of props]; metadata.dependencies [list of hooks/entities/services].\n"
        "• presentation/hook: metadata.rendering_mode ('client'); metadata.state [list of state keys]; metadata.actions [list of action names]; metadata.dependencies [list of stores/use cases/hooks].\n"
        "• application/store: metadata.state [list of state keys]; metadata.actions [list of action names]; metadata.dependencies [list of use case names].\n"
        "• application/use_case: metadata.inputs [list], metadata.outputs [list], metadata.dependencies [list of interfaces/entities/services].\n"
        "• application/interface: metadata.contract [list of method names], metadata.dependencies [list of entities/services].\n"
        "• infrastructure/adapter: metadata.implements [interface name]; metadata.methods [list of method names]; metadata.endpoint [string].\n"
        "• infrastructure/repository: metadata.implements [interface name]; metadata.methods [list of method names]; metadata.models [list of model names]. \n"
        "• infrastructure/models: metadata.fields [list of fields]; \n"
        "• domain/entity|service: metadata.fields [list]; metadata.methods [list of method names]; metadata.dependencies [list of dependencies].\n\n"
        "MANDATORY COMPLETENESS & LINKAGE:\n"
        "• Client components (rendering_mode == 'client') must list hooks in metadata.invocations; no use cases allowed.\n"
        "• Server components (rendering_mode == 'server') may only list use cases in metadata.invocations; no hooks allowed.\n"
        "• Hooks depend on stores/use cases declared in metadata.dependencies.\n"
        "• Use cases depend on interfaces/entities/services declared in metadata.dependencies.\n"
        "• Repositories/adapters must implement interfaces referenced by use cases.\n"
        "• Maintain dependency direction: presentation → application → domain.\n"
        "• Provide clear intent for each object in metadata.intent (1 sentence).\n\n"
        "DELIVERABLES:\n"
        "• Minimal set covering the flows (hooks/components + use cases + adapters/stores).\n"
        "• Output ONLY a JSON array. No prose. No null values; use {} or [].\n"
    )

    return Agent(
        model_name,
        output_type=List[Component],
        retries=3,
        model_settings=settings,
        system_prompt=prompt,
    )


def verify_components_agent(
    model_name: str = "openai:gpt-5-mini",
) -> Agent[List[Component]]:
    """Create an agent that consolidates and fixes Component definitions for consistency.

    Applies naming rules, dependency direction, and fills mandatory metadata.
    """
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="medium",
        openai_reasoning_summary="detailed",
        temperature=0.1,
    )
    prompt = (
        "You are a Clean Architecture reviewer. Consolidate and fix Component definitions to comply with the rules below.\n\n"
        "ENUMS (exact):\n"
        "• domain: domain/entity, domain/service\n"
        "• application: application/use_case, application/interface, application/store\n"
        "• infrastructure: infrastructure/adapter, infrastructure/repository, infrastructure/models\n"
        "• presentation: presentation/component, presentation/hook\n\n"
        "CHECKLIST:\n"
        "• All objects MUST have metadata.intent (one concise sentence).\n"
        "• Dependency direction: presentation → application → domain.\n"
        "• Naming consistent; remove duplicates names.\n"
        "• Rendering rules: components default to server; set to client if invocations include hook/store.\n"
        "• Infrastructure/adapter requires metadata.implements, metadata.endpoint, metadata.methods.\n"
        "• Infrastructure/repository requires metadata.implements, metadata.models, metadata.methods.\n"
        "• Merge overlapping definitions across features.\n\n"
        "COMPLETENESS FIXES:\n"
        "• Infer and fill minimal metadata (inputs, outputs, dependencies, state_fields, contract, endpoint, methods, models) when obvious.\n"
        "• Enforce naming: interfaces start with 'I' (e.g., IOrderAPI); adapters implement those interfaces.\n"
        "• Ensure linkage is closed: component → hook; hook → use cases; use_case → interfaces/stores; adapter/repository → interfaces.\n"
        "• Each component must retain its originating feature name in the `feature` field.\n"
        "• Client components (rendering_mode == 'client') must list hooks in metadata.invocations; no use cases allowed.\n"
        "• Server components (rendering_mode == 'server') may only list use cases in metadata.invocations; no hooks allowed.\n"
        "• Hooks depend on stores/use cases declared in metadata.dependencies.\n"
        "• Use cases depend on interfaces/entities/services declared in metadata.dependencies.\n"
        "• Repositories/adapters must implement interfaces referenced by use cases.\n"
        "• Keep schemas minimal; preserve intent + linkage.\n\n"
        "RETURN:\n"
        "• JSON array of Component (layer, type, name, properties, methods, metadata).\n"
    )
    return Agent(
        model_name,
        output_type=List[Component],
        retries=3,
        model_settings=settings,
        system_prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Parser v2
# ---------------------------------------------------------------------------


class ComponentParser:
    """Component Parser orchestrating PRD → features → components → verified set."""

    def __init__(self, model_name: str = "openai:gpt-5-mini") -> None:
        self.model_name = model_name
        self.prd_to_feature = prd_to_feature_agent(model_name)
        self.feature_to_objects = feature_to_component_agent(model_name)
        self.verify_objects = verify_components_agent(model_name)

    async def process(self, prd_text: str) -> str:
        """Run the full pipeline and return the verified components as JSON string."""
        try:
            # Step 1: Extract features from PRD
            logger.info("Extracting features from PRD text (%d chars)", len(prd_text))
            features: List[Feature] = (await self.prd_to_feature.run(prd_text)).output
            logger.info("Successfully extracted %d features", len(features))

            # Step 2: Generate components for each feature
            aggregated: List[Component] = []
            for idx, feature in enumerate(features, 1):
                logger.info(
                    "Processing feature %d/%d: %s", idx, len(features), feature.name
                )
                payload = json.dumps(
                    {"feature": feature.model_dump()}, ensure_ascii=False
                )
                result = (await self.feature_to_objects.run(payload)).output
                logger.info(
                    "Generated %d components for feature: %s", len(result), feature.name
                )
                aggregated.extend(result)

            logger.info("Total components generated: %d", len(aggregated))

            # Step 3: Verify and consolidate components
            logger.info("Verifying and consolidating components")
            verify_payload = json.dumps(
                {
                    "features": [
                        feature_item.model_dump() for feature_item in features
                    ],
                    "components": [
                        object_item.model_dump() for object_item in aggregated
                    ],
                },
                ensure_ascii=False,
            )
            verified_components: List[Component] = (
                await self.verify_objects.run(verify_payload)
            ).output
            logger.info(
                "Verification complete: %d final components", len(verified_components)
            )

            # Step 4: Create final output
            result = json.dumps(
                {
                    "features": [
                        feature_item.model_dump() for feature_item in features
                    ],
                    "components": [
                        object_item.model_dump() for object_item in verified_components
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            return result

        except Exception as e:
            logger.error("Error in component processing pipeline: %s", e)
            raise
