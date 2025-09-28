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
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FE Architecture schema (đồng bộ triết lý với BE)
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
    type: Optional[ComponentType] = Field(
        default=None,
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

    @model_validator(mode="after")
    def _enforce_layer_and_cleanup(self) -> "ArchitectureComponent":
        # type theo layer
        allowed_types = LAYER_ALLOWED_TYPES.get(self.layer, [])
        if self.type is None and allowed_types:
            self.type = allowed_types[0]
        elif self.type is not None and self.type not in allowed_types:
            raise ValueError(
                f"type '{self.type}' is not valid for layer '{self.layer}'. Allowed: {allowed_types}"
            )

        # name
        if isinstance(self.name, str):
            self.name = self.name.strip() or "UnnamedFEObject"

        # properties -> str:str
        cleaned_props: Dict[str, str] = {}
        for k, v in (self.properties or {}).items():
            if isinstance(k, str) and k.strip():
                cleaned_props[k.strip()] = v.strip() if isinstance(v, str) else str(v)
        self.properties = cleaned_props

        # methods -> list[str]
        cleaned_methods: List[str] = []
        for m in self.methods or []:
            if isinstance(m, str) and m.strip():
                cleaned_methods.append(m.strip())
        self.methods = cleaned_methods

        # metadata -> dict
        if not isinstance(self.metadata, dict):
            self.metadata = {}

        return self


# ---------------------------------------------------------------------------
# Helper: normalize FE object before validation
# ---------------------------------------------------------------------------


def _apply_component_defaults(raw_obj: Dict[str, Any]) -> Dict[str, Any]:
    obj: Dict[str, Any] = dict(raw_obj)

    # name
    name_value = obj.get("name")
    if not isinstance(name_value, str) or not name_value.strip():
        obj["name"] = "UnnamedFEObject"

    # properties/methods
    obj["properties"] = (
        obj.get("properties") if isinstance(obj.get("properties"), dict) else {}
    )
    obj["methods"] = obj.get("methods") if isinstance(obj.get("methods"), list) else []

    # metadata
    metadata_value = obj.get("metadata")
    metadata = metadata_value if isinstance(metadata_value, dict) else {}

    # Normalize invocations metadata (defer attaching to metadata until type known)
    invocations_raw = metadata.get("invocations")
    normalized_invocations: List[Dict[str, str]] = []

    def _append_invocation(inv_type: str, value: Any) -> None:
        if isinstance(inv_type, str) and isinstance(value, str):
            inv_type = inv_type.strip()
            value = value.strip()
            if inv_type and value:
                normalized_invocations.append({"type": inv_type, "name": value})

    if isinstance(invocations_raw, list):
        for item in invocations_raw:
            if isinstance(item, dict):
                _append_invocation(item.get("type", ""), item.get("name", ""))
            elif isinstance(item, str):
                _append_invocation("unknown", item)

    # type/layer normalization
    type_value = obj.get("type")
    type_str = type_value.strip() if isinstance(type_value, str) else ""
    layer_value = obj.get("layer")
    layer_str = layer_value.strip() if isinstance(layer_value, str) else ""

    all_types = {
        allowed_type
        for allowed_types in LAYER_ALLOWED_TYPES.values()
        for allowed_type in allowed_types
    }

    if type_str not in all_types:
        short_map: Dict[str, str] = {
            "entity": "domain/entity",
            "service": "domain/service",
            "services": "domain/service",
            "use_case": "application/use_case",
            "usecase": "application/use_case",
            "interface": "application/interface",
            "store": "application/store",
            "repository": "infrastructure/repository",
            "repo": "infrastructure/repository",
            "adapter": "infrastructure/adapter",
            "component": "presentation/component",
            "hook": "presentation/hook",
        }
        lower_type = (
            (type_value or "").strip().replace(" ", "_").lower()
            if isinstance(type_value, str)
            else ""
        )
        type_str = short_map.get(lower_type, "")

    if not layer_str and type_str:
        layer_str = type_str.split("/", 1)[0]

    if layer_str not in LAYER_ALLOWED_TYPES:
        # Try infer layer from type if possible, otherwise default to application (stricter than forcing presentation)
        inferred_layer = type_str.split("/", 1)[0] if "/" in type_str else ""
        layer_str = (
            inferred_layer if inferred_layer in LAYER_ALLOWED_TYPES else "application"
        )

    # Heuristic from name if still missing
    if not type_str or type_str not in all_types:
        name_hint = obj.get("name") if isinstance(obj.get("name"), str) else ""
        if name_hint.endswith("Component"):
            type_str = "presentation/component"
            layer_str = "presentation"
        elif name_hint.endswith("Hook"):
            type_str = "presentation/hook"
            layer_str = "presentation"
        elif name_hint.endswith("Store"):
            type_str = "application/store"
            layer_str = "application"
        elif name_hint.endswith("UseCase"):
            type_str = "application/use_case"
            layer_str = "application"
        elif name_hint.endswith("Repository"):
            type_str = "infrastructure/repository"
            layer_str = "infrastructure"
        elif name_hint.endswith("Adapter"):
            type_str = "infrastructure/adapter"
            layer_str = "infrastructure"
        elif name_hint.endswith("Entity"):
            type_str = "domain/entity"
            layer_str = "domain"
        elif name_hint.endswith("Service"):
            type_str = "domain/service"
            layer_str = "domain"

    # Guard final layer/type
    allowed_types_for_layer = LAYER_ALLOWED_TYPES[layer_str]
    if not type_str:
        type_str = allowed_types_for_layer[0]
    elif type_str not in allowed_types_for_layer:
        type_str = allowed_types_for_layer[0]

    # Set layer/type after normalization
    obj["layer"] = layer_str
    obj["type"] = type_str

    # Defaults metadata theo type
    if type_str == "presentation/component":
        metadata.setdefault("props", {})
        # Merge normalized_invocations into metadata.invocations (component only), with dedupe by (type,name)
        existing_invs = metadata.get("invocations", [])
        if not isinstance(existing_invs, list):
            existing_invs = []
        seen = {
            (i.get("type"), i.get("name")) for i in existing_invs if isinstance(i, dict)
        }
        for inv in normalized_invocations:
            key = (inv.get("type"), inv.get("name"))
            if key not in seen:
                existing_invs.append(inv)
                seen.add(key)
        metadata["invocations"] = existing_invs

        # rendering mode
        if not isinstance(metadata.get("rendering_mode"), str):
            metadata["rendering_mode"] = "server"
        else:
            metadata["rendering_mode"] = (
                metadata["rendering_mode"].strip().lower() or "server"
            )
        # Auto-set rendering_mode to client if invoking hooks/stores
        if any(
            inv.get("type") in {"hook", "store"}
            for inv in metadata.get("invocations", [])
        ):
            metadata["rendering_mode"] = "client"
    elif type_str == "presentation/hook":
        metadata.setdefault("state_fields", [])
        metadata["rendering_mode"] = "client"
    elif type_str == "application/store":
        # keep lightweight defaults
        metadata.setdefault("state_fields", [])
        metadata.setdefault("dependencies", [])
    elif type_str == "application/use_case":
        metadata.setdefault("inputs", [])
        metadata.setdefault("outputs", [])
        metadata.setdefault("dependencies", [])
    elif type_str == "application/interface":
        metadata.setdefault("contract", [])  # list of method names
        # If contract is empty but methods exist, mirror methods into contract
        if (not metadata.get("contract")) and isinstance(obj.get("methods"), list):
            inferred_contract = [
                m.strip()
                for m in obj.get("methods", [])
                if isinstance(m, str) and m.strip()
            ]
            if inferred_contract:
                metadata["contract"] = inferred_contract
    elif type_str.startswith("domain/"):
        # optional: expose fields inferred from properties if any
        if obj.get("properties"):
            metadata.setdefault("fields", obj["properties"])  # lightweight hint
    elif type_str == "infrastructure/repository":
        metadata.setdefault("implements", "")
        # Set useful defaults if missing/empty
        if not isinstance(metadata.get("method"), str) or not metadata.get("method"):
            metadata["method"] = "GET"
        if not isinstance(metadata.get("endpoint"), str) or not metadata.get(
            "endpoint"
        ):
            metadata["endpoint"] = "/"
    elif type_str == "infrastructure/adapter":
        metadata.setdefault("implements", "")
        if not isinstance(metadata.get("method"), str) or not metadata.get("method"):
            metadata["method"] = "GET"
        if not isinstance(metadata.get("endpoint"), str) or not metadata.get(
            "endpoint"
        ):
            metadata["endpoint"] = "/"

    # Ensure every object has a minimal intent sentence
    if (
        not isinstance(metadata.get("intent"), str)
        or not metadata.get("intent").strip()
    ):
        base_name = obj.get("name") if isinstance(obj.get("name"), str) else "Object"
        if type_str == "presentation/component":
            metadata["intent"] = (
                f"Render UI and orchestrate interactions for {base_name}."
            )
        elif type_str == "presentation/hook":
            metadata["intent"] = (
                f"Provide client-side state and actions for {base_name}."
            )
        elif type_str == "application/use_case":
            metadata["intent"] = f"Execute application logic for {base_name}."
        elif type_str == "application/interface":
            metadata["intent"] = f"Define contract for {base_name}."
        elif type_str == "application/store":
            metadata["intent"] = f"Hold application state for {base_name}."
        elif type_str.startswith("infrastructure/"):
            metadata["intent"] = f"Integrate external concerns for {base_name}."
        elif type_str.startswith("domain/"):
            metadata["intent"] = f"Encapsulate domain rules for {base_name}."

    obj["metadata"] = metadata
    return obj


# ---------------------------------------------------------------------------
# PydanticAI pipelines (agents)
# ---------------------------------------------------------------------------


class FrontendFeature(BaseModel):
    """Frontend feature specification extracted from PRD (framework-agnostic).

    Fields summarize the user goal, actors, key UI steps, and application
    touchpoints to drive FE object generation in later steps.
    """

    name: str = Field(description="Tên feature PascalCase")
    actor: str = Field(description="Vai trò người dùng chính")
    project_goal: str = Field(description="Mục tiêu business của feature")
    ui_flow: List[str] = Field(description="Chuỗi bước UI quan trọng")
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
        "You are a Frontend Requirements Analyst."
        "Extract frontend features for a generic Clean Architecture-based frontend codebase (framework-agnostic).\n\n"
        "RETURN FORMAT:\n"
        "• list of {name, actor, project_goal, ui_flow, application_touchpoints, noteworthy_constraints}.\n\n"
        "GUIDELINES:\n"
        "• Each feature = distinct user goal requiring orchestration.\n"
        "• Keep ui_flow at high level (3–7 steps max).\n"
        "• application_touchpoints should name generic Clean Architecture entities (e.g., AuthUseCase, CartStore).\n"
        "• noteworthy_constraints may include UX rules, security requirements, performance considerations.\n"
        "• Generate key features only.\n"
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
        "FIELD RULES (keep minimal, framework-agnostic):\n"
        "• All objects MUST have `metadata.intent`.\n"
        "• presentation/component: metadata.rendering_mode ('server' | 'client'); metadata.invocations [list of {type, name}]; metadata.props {}.\n"
        "• presentation/hook: metadata.rendering_mode must be 'client'; metadata.invocations [list of {type, name}] conveying stores/use cases/hooks; optional state_fields [list].\n"
        "• application/store: metadata.state_fields [list of keys]; metadata.dependencies [list of use case names]; metadata.invocations optional for helper calls.\n"
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
        "• Provide clear intent for each object in metadata.intent (1 sentence).\n"
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

        aggregated: List[Dict[str, Any]] = []
        for feature in features:
            logger.info("Generating FE objects for feature: %s", feature.name)
            payload = json.dumps({"feature": feature.model_dump()}, ensure_ascii=False)
            result = (await self.feature_to_objects.run(payload)).output

            if not result:
                logger.warning("No FE objects for feature %s; skip", feature.name)
                continue

            batch: List[Dict[str, Any]] = []
            for idx, item in enumerate(result):
                if item is None:
                    logger.warning(
                        "Null FE object at #%s for %s; skip", idx, feature.name
                    )
                    continue
                if isinstance(item, ArchitectureComponent):
                    item_dict = item.model_dump()
                elif isinstance(item, dict):
                    item_dict = item
                else:
                    logger.warning(
                        "Unexpected FE object type %s; skip", type(item).__name__
                    )
                    continue

                normalized = _apply_component_defaults(item_dict)
                try:
                    validated = ArchitectureComponent.model_validate(normalized)
                    batch.append(
                        validated.model_dump()
                        if isinstance(validated, ArchitectureComponent)
                        else normalized
                    )
                except ValidationError as exc:
                    logger.warning(
                        "Validation failed after normalization for %s: %s",
                        normalized.get("name"),
                        exc,
                    )
                    batch.append(normalized)

            if not batch:
                logger.warning("All FE objects discarded for feature %s", feature.name)
                continue
            aggregated.extend(batch)

        logger.info("Verifying FE objects")
        verify_payload = json.dumps(
            {
                "features": [f.model_dump() for f in features],
                "fe_objects": aggregated,
            },
            ensure_ascii=False,
        )
        verified = (await self.verify_objects.run(verify_payload)).output

        final_objects: List[Dict[str, Any]] = []
        if isinstance(verified, list):
            for idx, obj in enumerate(verified):
                if obj is None:
                    logger.warning("Verified FE object #%s is null; discarded", idx)
                    continue
                if isinstance(obj, ArchitectureComponent):
                    obj_dict = obj.model_dump()
                elif isinstance(obj, dict):
                    obj_dict = obj
                else:
                    logger.warning(
                        "Unexpected verified object type %s; discarded",
                        type(obj).__name__,
                    )
                    continue

                obj_dict = _apply_component_defaults(obj_dict)
                try:
                    validated = ArchitectureComponent.model_validate(obj_dict)
                    final_objects.append(
                        validated.model_dump()
                        if isinstance(validated, ArchitectureComponent)
                        else obj_dict
                    )
                except ValidationError as exc:
                    logger.warning(
                        "Verified validation failed for %s: %s",
                        obj_dict.get("name"),
                        exc,
                    )
                    final_objects.append(obj_dict)
        else:
            logger.warning("Verifier output is not list; skip finalization")

        return json.dumps(final_objects, ensure_ascii=False)


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
