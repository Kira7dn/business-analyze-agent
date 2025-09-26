#!/usr/bin/env python3
"""Frontend Object Parser pipeline using Pydantic AI."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import jsonschema

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

# -----------------------------
# FEObject Schema
# -----------------------------
class FEObject(BaseModel):
    layer: str = Field(..., description="domain | application | infrastructure | presentation")
    type: str = Field(..., description="entity | service | interface | use_case | repository | hook | component")
    name: str = Field(..., description="PascalCase name for class/hook/component")
    properties: Optional[Dict[str, str]] = Field(default_factory=dict)
    methods: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


VALID_LAYERS = {"domain", "application", "infrastructure", "presentation"}
VALID_TYPES = {"entity", "service", "interface", "use_case", "repository", "hook", "component"}


def validate_feobject(obj: Dict[str, Any]) -> FEObject:
    """Validate dict -> FEObject (raises ValidationError on failure)."""
    if "layer" not in obj or obj["layer"] not in VALID_LAYERS:
        raise ValidationError([{"loc": ("layer",), "msg": f"layer must be one of {VALID_LAYERS}", "type": "value_error"}], FEObject)
    if "type" not in obj or obj["type"] not in VALID_TYPES:
        raise ValidationError([{"loc": ("type",), "msg": f"type must be one of {VALID_TYPES}", "type": "value_error"}], FEObject)
    if "name" not in obj or not isinstance(obj["name"], str) or not obj["name"].strip():
        raise ValidationError([{"loc": ("name",), "msg": "name is required and must be non-empty string", "type": "value_error"}], FEObject)
    return FEObject(**obj)




# -----------------------------
# PydanticAI pipeline cho FE objects
# -----------------------------


class FrontendFeature(BaseModel):
    """Mô tả một tính năng frontend từ yêu cầu PRD."""

    name: str = Field(..., description="Tên feature ở dạng PascalCase, phản ánh user goal")
    actor: str = Field(..., description="Người dùng chính tương tác feature")
    project_goal: str = Field(..., description="Tóm tắt mục tiêu business liên quan feature")
    ui_flow: List[str] = Field(..., description="Chuỗi bước UI hoặc interaction quan trọng")
    application_touchpoints: List[str] = Field(
        default_factory=list,
        description="Các use case / interface phía application layer mà UI cần gọi",
    )
    noteworthy_constraints: List[str] = Field(
        default_factory=list,
        description="Các constraint quan trọng: UX, performance, accessibility",
    )


class FrontendFeatureOutput(BaseModel):
    project_description: str = Field(..., description="Mô tả tổng quan dự án từ PRD")
    features: List[FrontendFeature] = Field(..., description="Danh sách feature frontend")


class FeatureFEObjects(BaseModel):
    feature: str = Field(..., description="Tên feature đang chuyển đổi")
    rationale: str = Field(..., description="Giải thích mapping giữa feature và FEObjects")
    fe_objects: List[FEObject] = Field(..., description="Danh sách FEObject đã tạo cho feature")


def prd_to_frontend_feature_agent(model_name: str = "openai:o4-mini") -> Agent[FrontendFeatureOutput]:
    """Tạo agent phân tích PRD thành danh sách feature frontend."""

    settings_ai = OpenAIResponsesModelSettings(
        openai_reasoning_effort="low",
        openai_reasoning_summary="detailed",
        temperature=0.2,
    )

    prompt = (
        "You are a Frontend Requirements Analyst following the workflow described in workflowE.md. "
        "Extract frontend features that will drive Clean Architecture deliverables for a Next.js + TypeScript codebase.\n\n"
        "RETURN FORMAT:\n"
        "• project_description: concise summary of business goals from PRD.\n"
        "• features: list where each item contains name, actor, project_goal, ui_flow, application_touchpoints, noteworthy_constraints.\n\n"
        "GUIDELINES:\n"
        "• Features must map to distinct user goals or flows that require frontend orchestration.\n"
        "• ui_flow should list key UI interactions (view, input, submit, navigate).\n"
        "• application_touchpoints references expected application layer interactions (use cases, stores, interfaces).\n"
        "• Consider Clean Architecture dependency rule: presentation → application → domain.\n"
        "• Include UX constraints, performance SLAs, accessibility notes when mentioned.\n"
    )

    return Agent(
        model_name,
        output_type=FrontendFeatureOutput,
        retries=3,
        model_settings=settings_ai,
        system_prompt=prompt,
    )


def feature_to_fe_objects_agent(model_name: str = "openai:o4-mini") -> Agent[FeatureFEObjects]:
    """Tạo agent chuyển từng feature thành FE objects theo chuẩn Clean Architecture."""

    settings_ai = OpenAIResponsesModelSettings(
        openai_reasoning_effort="low",
        openai_reasoning_summary="detailed",
        temperature=0.15,
    )

    prompt = (
        "You are a Frontend Architect. Given a single feature context, produce FEObject definitions covering presentation/application/domain/infrastructure layers per workflowE.md.\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "• fe_objects must form a minimal yet complete slice supporting the feature.\n"
        "• Ensure naming consistency (PascalCase). Hooks/components in presentation must call application use cases or stores explicitly.\n"
        "• Application layer should expose use cases, interfaces, stores (Zustand) where needed. For each use case include metadata.inputs (DTO fields) and metadata.outputs (response shape).\n"
        "• Domain layer only when frontend needs shared entities/models; include properties with field:type pairs.\n"
        "• Infrastructure layer optional: only if frontend manages adapters (e.g., HTTP client wrappers). For each adapter include metadata.endpoint, metadata.method, metadata.payload_schema.\n"
        "• Hooks must declare metadata.stores_used, metadata.use_cases_called, and metadata.state_fields.\n"
        "• Components must declare metadata.props (prop:type) và metadata.hooks_used.\n"
        "• Stores must include metadata.state_schema and metadata.actions with payload schema.\n\n"
        "RETURN JSON with fields feature, rationale, fe_objects. Each fe_object must include properties/methods/metadata filled appropriately.\n"
    )

    return Agent(
        model_name,
        output_type=FeatureFEObjects,
        retries=3,
        model_settings=settings_ai,
        system_prompt=prompt,
    )


def verify_fe_objects_agent(model_name: str = "openai:o4-mini") -> Agent[List[FEObject]]:
    """Tạo agent hợp nhất & kiểm tra FEObject theo chuẩn workflowE."""

    settings_ai = OpenAIResponsesModelSettings(
        openai_reasoning_effort="low",
        openai_reasoning_summary="detailed",
        temperature=0.1,
    )

    prompt = (
        "You are a Frontend Clean Architecture reviewer. Consolidate and fix FEObject definitions to comply with workflowE.md.\n\n"
        "CHECKLIST:\n"
        "• Layers follow dependency direction (presentation → application → domain).\n"
        "• Naming consistent, no duplicates, remove numeric/generic names.\n"
        "• Reject any fe_object lacking properties/metadata when required. Enrich metadata with contract details.\n"
        "• Hooks must list metadata.stores_used & metadata.use_cases_called. Components must expose metadata.props & metadata.hooks_used.\n"
        "• Application layer objects must list metadata.inputs, metadata.outputs, metadata.dependencies (interfaces/stores).\n"
        "• Domain entities/services must declare properties & methods representing business logic.\n"
        "• Infrastructure entries must specify metadata.endpoint/method or metadata.client used.\n"
        "• Merge overlapping definitions across features, ensuring minimal consistent FE object set.\n"
    )

    return Agent(
        model_name,
        output_type=List[FEObject],
        retries=3,
        model_settings=settings_ai,
        system_prompt=prompt,
    )


class FEObjectParser:
    """Pipeline parser chuyển PRD -> FE objects."""

    def __init__(self, model_name: str = "openai:o4-mini") -> None:
        self.model_name = model_name
        self.prd_to_feature = prd_to_frontend_feature_agent(model_name)
        self.feature_to_objects = feature_to_fe_objects_agent(model_name)
        self.verify_objects = verify_fe_objects_agent(model_name)

    async def process(self, prd_text: str) -> str:
        """Chạy pipeline và trả về JSON chuỗi FEObject đã verify."""

        feature_output: FrontendFeatureOutput = (
            await self.prd_to_feature.run(prd_text)
        ).output
        features = feature_output.features
        project_description = feature_output.project_description

        os.makedirs("projects/warehouse/frontend", exist_ok=True)
        with open(
            "projects/warehouse/frontend/features.json",
            "w",
            encoding="utf-8",
        ) as features_file:
            json.dump(feature_output.model_dump(), features_file, ensure_ascii=False, indent=2)

        print(f"Generated {len(features)} frontend features")

        feature_objects: List[Dict[str, Any]] = []
        aggregated_objects: List[Dict[str, Any]] = []

        for feature in features:
            print(f"Generating FE objects for feature: {feature.name}")
            payload = json.dumps(
                {
                    "project_description": project_description,
                    "feature": feature.model_dump(),
                },
                ensure_ascii=False,
            )
            result: FeatureFEObjects = (await self.feature_to_objects.run(payload)).output
            result_dict = result.model_dump()
            feature_objects.append(result_dict)
            aggregated_objects.extend(result_dict["fe_objects"])

        with open(
            "projects/warehouse/frontend/feature_fe_objects.json",
            "w",
            encoding="utf-8",
        ) as feature_objects_file:
            json.dump(feature_objects, feature_objects_file, ensure_ascii=False, indent=2)

        verify_payload = json.dumps(
            {
                "project_description": project_description,
                "features": [feature.model_dump() for feature in features],
                "fe_objects": aggregated_objects,
            },
            ensure_ascii=False,
        )

        verified_objects = (
            await self.verify_objects.run(verify_payload)
        ).output
        verified_objects = [obj.model_dump() for obj in verified_objects]

        schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "fe_schema.json")
        try:
            with open(schema_path, "r", encoding="utf-8") as schema_file:
                schema = json.load(schema_file)
        except OSError as exc:
            print(f"[WARN] Unable to load schema at {schema_path}: {exc}")
            schema = None

        if schema:
            for obj in verified_objects:
                try:
                    jsonschema.validate(instance=obj, schema=schema)
                except jsonschema.ValidationError as exc:
                    print(f"[WARN] FEObject validation failed for {obj.get('name')}: {exc.message}")

        with open(
            "projects/warehouse/frontend/verified_fe_objects.json",
            "w",
            encoding="utf-8",
        ) as verified_file:
            json.dump(verified_objects, verified_file, ensure_ascii=False, indent=2)

        print(f"Verified {len(verified_objects)} FE objects")

        return json.dumps(verified_objects, ensure_ascii=False)


DEFAULT_PRD_PATH = "projects/warehouse/PRD.md"
DEFAULT_OUTPUT_PATH = "projects/warehouse/fe_objects.json"


async def main() -> None:
    """Main entry point để chạy FE object parser."""

    if not os.path.exists(DEFAULT_PRD_PATH):
        print(f"PRD file not found: {DEFAULT_PRD_PATH}")
        return

    with open(DEFAULT_PRD_PATH, "r", encoding="utf-8") as prd_file:
        prd_text = prd_file.read()

    parser = FEObjectParser()
    print("\n=== FE Object Parser ===")
    fe_objects_json = await parser.process(prd_text)

    with open(DEFAULT_OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        output_file.write(fe_objects_json)

    print(f"Final output saved to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
