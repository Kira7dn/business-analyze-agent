"""
Backend Object Parser: LLM pipeline to convert PRD -> User Flow -> Clean Architecture classes.

This module uses PydanticAI to define agents that process a Product Requirements Document
through multiple transformation stages to generate backend class definitions.

The pipeline follows these steps:
1. Convert PRD text to UserFlow objects
2. Convert each UserFlow directly into Clean Architecture class definitions
3. Chuẩn hoá và xác thực kết quả bằng các `BaseModel` của Pydantic
"""

from typing import Any, Dict, List, Literal, Optional
import json
import logging
import sys
import os

from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain class schema
# ---------------------------------------------------------------------------

LayerName = Literal["domain", "application", "infrastructure", "presentation"]

ComponentType = Literal[
    "domain/entity",
    "domain/service",
    "application/interface",
    "application/use_case",
    "infrastructure/model",
    "infrastructure/repository",
    "infrastructure/adapter",
    "presentation/schema",
    "presentation/dependency",
    "presentation/router",
]

LAYER_ALLOWED_TYPES: Dict[LayerName, List[ComponentType]] = {
    "domain": ["domain/entity", "domain/service"],
    "application": ["application/interface", "application/use_case"],
    "infrastructure": [
        "infrastructure/model",
        "infrastructure/repository",
        "infrastructure/adapter",
    ],
    "presentation": [
        "presentation/schema",
        "presentation/dependency",
        "presentation/router",
    ],
}


class MethodDefinition(BaseModel):
    """Function/backend interface specification for a data flow."""

    method_name: str = Field(description="Name of the function in snake_case format.")
    description: str = Field(
        description="Concise description of the function's purpose."
    )
    parameters: List[str] = Field(
        description="List of parameters in format 'name: type' (e.g., 'user_id: int')."
    )
    return_type: str = Field(
        description="Return type of the function (e.g., 'dict', 'User', 'bool')."
    )


class ArchitectureComponent(BaseModel):
    """Class definition for a feature."""

    name: str = Field(description="Name of the class in PascalCase format.")
    type: Optional[ComponentType] = Field(
        default=None,
        description=(
            "Architectural classification tag mirroring layer (e.g., 'domain/entity', 'application/use_case')."
        ),
    )
    layer: LayerName = Field(
        description=(
            "Top-level Clean Architecture layer tag. One of: "
            "'domain', 'application', 'infrastructure', 'presentation'."
        )
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description=(
            "List of interface names this class depends on (e.g., "
            "['IOrderRepository', 'IPaymentAdapter'])."
        ),
    )
    properties: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional key:value map of property name to type.",
    )
    description: str = Field(
        description=(
            "Description of the class's purpose and role within the " "architecture."
        )
    )
    attributes: List[str] = Field(
        default_factory=list,
        description="List of attributes in format 'name: type'.",
    )
    methods: List[MethodDefinition] = Field(
        default_factory=list,
        description=("List of method definitions (name, parameters, description)."),
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured information (contracts, DTOs, endpoint details, etc.).",
    )

    @model_validator(mode="after")
    def _enforce_layer_and_cleanup(self) -> "ArchitectureComponent":
        allowed_types = LAYER_ALLOWED_TYPES.get(self.layer, [])
        if self.type is None and allowed_types:
            self.type = allowed_types[0]
        elif self.type is not None and self.type not in allowed_types:
            raise ValueError(
                f"type '{self.type}' is not valid for layer '{self.layer}'. Allowed: {allowed_types}"
            )

        self.dependencies = [
            dep.strip()
            for dep in self.dependencies
            if isinstance(dep, str) and dep.strip()
        ]

        self.attributes = [
            attr.strip()
            for attr in self.attributes
            if isinstance(attr, str) and attr.strip()
        ]

        cleaned_methods: List[MethodDefinition] = []
        for method in self.methods:
            if isinstance(method, MethodDefinition):
                cleaned_methods.append(method)
            elif isinstance(method, dict):
                cleaned_methods.append(MethodDefinition(**method))

        if not isinstance(self.metadata, dict):
            self.metadata = {}

        return self


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _apply_component_defaults(cls: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize backend class dictionaries before validation."""

    obj: Dict[str, Any] = dict(cls)

    name_value = obj.get("name")
    if not isinstance(name_value, str) or not name_value.strip():
        obj["name"] = "UnnamedClass"

    properties_value = obj.get("properties")
    obj["properties"] = properties_value if isinstance(properties_value, dict) else {}

    dependencies_value = obj.get("dependencies")
    if isinstance(dependencies_value, list):
        obj["dependencies"] = [
            dep.strip()
            for dep in dependencies_value
            if isinstance(dep, str) and dep.strip()
        ]
    else:
        obj["dependencies"] = []

    attributes_value = obj.get("attributes")
    if isinstance(attributes_value, list):
        obj["attributes"] = [
            attr.strip()
            for attr in attributes_value
            if isinstance(attr, str) and attr.strip()
        ]
    else:
        obj["attributes"] = []

    methods_value = obj.get("methods")
    normalized_methods: List[Dict[str, Any]] = []
    if isinstance(methods_value, list):
        for method in methods_value:
            if isinstance(method, MethodDefinition):
                normalized_methods.append(method.model_dump())
            elif isinstance(method, dict):
                normalized_methods.append(method)
    obj["methods"] = normalized_methods

    layer_value = obj.get("layer")
    layer_str = layer_value.strip() if isinstance(layer_value, str) else ""
    type_value = obj.get("type")
    type_str = type_value.strip() if isinstance(type_value, str) else ""

    inferred_layer = ""
    if type_str and "/" in type_str:
        inferred_layer = type_str.split("/", 1)[0]
    elif layer_str:
        inferred_layer = layer_str.split("/", 1)[0] if "/" in layer_str else layer_str

    if inferred_layer not in LAYER_ALLOWED_TYPES:
        inferred_layer = "application"

    allowed_types = LAYER_ALLOWED_TYPES[inferred_layer]
    if not type_str or type_str not in allowed_types:
        type_str = allowed_types[0]

    obj["layer"] = inferred_layer
    obj["type"] = type_str

    metadata_value = obj.get("metadata")
    metadata = metadata_value if isinstance(metadata_value, dict) else {}
    metadata.setdefault("top_layer", inferred_layer)

    def _attributes_to_field_map(attributes: List[str]) -> Dict[str, str]:
        field_map: Dict[str, str] = {}
        for attr in attributes:
            if isinstance(attr, str) and ":" in attr:
                name, type_hint = attr.split(":", 1)
                field_map[name.strip()] = type_hint.strip()
        return field_map

    if type_str.startswith("domain/"):
        metadata.setdefault("fields", _attributes_to_field_map(obj["attributes"]))

    if type_str.startswith("presentation/schema"):
        metadata.setdefault("fields", _attributes_to_field_map(obj["attributes"]))

    if type_str == "application/interface":
        metadata.setdefault(
            "contract",
            [
                {
                    "method": (
                        f"{method.get('method_name')}"
                        f"({', '.join(method.get('parameters', []))})"
                        f" -> {method.get('return_type', 'None')}"
                    )
                }
                for method in obj["methods"]
                if isinstance(method, dict)
            ],
        )

    if type_str == "application/use_case":
        metadata.setdefault("inputs", "")
        metadata.setdefault("outputs", "")
        metadata.setdefault("dependencies", obj["dependencies"])
        metadata.setdefault("steps", [])

    if type_str.startswith("infrastructure/repository"):
        metadata.setdefault("storage", {"uses": "SQLAlchemy", "model": ""})

    if type_str.startswith("presentation/dependency"):
        metadata.setdefault(
            "providers",
            [
                method.get("method_name")
                for method in obj["methods"]
                if isinstance(method, dict) and method.get("method_name")
            ],
        )

    obj["metadata"] = metadata

    return obj


# ---------------------------------------------------------------------------
# PydanticAI pipeline cho BE objects
# ---------------------------------------------------------------------------


class UserFlow(BaseModel):
    """User flow for a feature."""

    feature: str = Field(
        description="Name of the feature in PascalCase action-oriented format (e.g., 'RegisterUser', 'SearchProduct'). This will become the Use Case class name."
    )
    actor: str = Field(
        description="Primary user or system actor for this flow (e.g., 'Customer', 'Admin')."
    )
    steps: List[str] = Field(
        description="Ordered list of steps the actor takes to complete this flow."
    )


class UserFlowOutput(BaseModel):
    """Collection of user flows extracted from PRD."""

    userflows: List[UserFlow] = Field(
        description="List of all user flows extracted from the PRD."
    )
    project_description: str = Field(description="Project description.")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional structured information (contracts, DTOs, endpoint details, etc.).",
    )


def prd_to_flow_agent(model_name: str = "openai:o4-mini"):
    """
    Create an agent to convert PRD to a UserFlowList.

    Args:
        model_name: The name of the LLM model to use

    Returns:
        Agent configured to convert PRD to UserFlowList
    """
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="low",
        openai_reasoning_summary="detailed",
        temperature=0.2,
    )
    return Agent(
        model_name,
        output_type=UserFlowOutput,
        retries=3,
        model_settings=settings,
        system_prompt=(
            "You are a requirements analyst specializing in Clean Architecture feature extraction. Extract UserFlows that will map 1:1 to Use Cases.\n\n"
            "EXTRACTION RULES:\n"
            "• Each feature becomes exactly ONE Use Case with single execute() method\n"
            "• Feature name should be PascalCase and action-oriented (e.g., 'PlaceOrder', 'RegisterUser')\n"
            "• Actor represents the primary user role for this workflow\n"
            "• Steps should be user actions that trigger backend operations\n\n"
            "STEP GUIDELINES:\n"
            "• Focus on user interactions that require backend processing\n"
            "• Avoid UI-only steps (e.g., 'User clicks button')\n"
            "• Include validation and business logic steps\n"
            "• Each step should trigger a backend operation\n"
            "• Keep steps sequential and necessary only\n\n"
            "OUTPUT REQUIREMENTS:\n"
            "1. Extract ALL distinct features from PRD sections:\n"
            "   - Feature Suggestions by Role\n"
            "   - Functional Requirements\n"
            "   - User Stories\n"
            "2. Create UserFlow for each feature with:\n"
            "   - feature: PascalCase action name\n"
            "   - actor: Primary user role\n"
            "   - steps: Sequential user actions requiring backend\n"
            "3. Include project_description from PRD overview\n\n"
            "Each UserFlow will become a Use Case in Clean Architecture."
        ),
    )


def flow_to_component_agent(model_name: str = "openai:o4-mini"):
    """
    Create an agent to convert UserFlow context directly into Clean Architecture classes.

    Args:
        model_name: The name of the LLM model to use

    Returns:
        Agent configured to convert UserFlow to ClassOutput
    """
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="low",
        openai_reasoning_summary="detailed",
        temperature=0.1,
    )
    return Agent(
        model_name,
        output_type=List[ArchitectureComponent],
        retries=3,
        model_settings=settings,
        system_prompt=(
            "You are a backend architect specializing in Onion/Clean Architecture. Given project description and a single User Flow (feature, actor, sequential steps), design the minimal set of classes required to fulfill that flow.\n\n"
            "CRITICAL RULES:\n"
            "1. Inner layers must NOT depend on outer layers (inward dependency rule).\n"
            "2. Presentation depends on Application; Application depends on Domain; Infrastructure implements Application interfaces.\n"
            "3. Domain entities have NO I/O operations (no save, fetch, etc.).\n"
            "4. Application layer depends ONLY on interfaces (ports), never implementations.\n"
            "5. Infrastructure implements Application interfaces.\n\n"
            "FORMAT:\n"
            "- name: PascalCase\n"
            "- attributes/methods: snake_case\n"
            "- dependencies: list of interface names this class depends on (e.g., ['IOrderRepository', 'IPaymentAdapter'])\n"
            "- layer: exact tags from: 'domain/entity', 'domain/service', 'application/interface', 'application/use_case', 'infrastructure/model', 'infrastructure/repository', 'infrastructure/adapter', 'presentation/schema', 'presentation/router', 'presentation/dependency'\n\n"
            "LAYER SPECIFICATIONS:\n"
            "Domain Layer:\n"
            "• Entities (domain/entity): Business objects with intrinsic behaviors only. NO persistence methods.\n"
            "• Services (domain/service): Pure business algorithms. May depend on other domain services.\n"
            "Application Layer:\n"
            "• Use Case (application/use_case): Orchestrates flow steps via interfaces. Derive execute() parameters from User Flow inputs, outputs from effects. Dependencies are injected interfaces.\n"
            "  IMPORTANT: Method descriptions must include detailed step-by-step execution logic:\n"
            "  - Validation steps (what to validate and how)\n"
            "  - Domain method calls (which entity/service methods to call)\n"
            "  - Repository interactions (fetch, save operations)\n"
            "  - Business logic sequence (order of operations)\n"
            "  - Error handling scenarios\n"
            "  Example: 'Validates request.items not empty, creates Order entity via Order.create(), validates business rules via order.validate_new(), saves via repository.save(), logs audit via audit_repository.log(), returns CreateOrderResponse with order.id'\n"
            "• Interface (application/interface): Abstract ports for repositories/adapters. metadata.contract must document every method signature.\n"
            "Infrastructure Layer:\n"
            "• Repository (infrastructure/repository): Implements application interfaces. metadata.storage or metadata.endpoint/method/payload_schema must be populated.\n"
            "• Adapter (infrastructure/adapter): Implements application interfaces, metadata must describe external capability (endpoint, method, payload).\n"
            "• Model (infrastructure/model): ORM/database mapping classes. Provide properties or metadata.fields mapping persistence columns.\n"
            "Presentation Layer:\n"
            "• Schema (presentation/schema): Request/Response DTOs. metadata.fields must enumerate fields with types.\n"
            "• Router (presentation/router): HTTP endpoint handlers. Each method accepts Request DTO + repository providers as parameters, instantiates Use Case internally, calls execute(request), returns Response DTO. Do NOT store Use Cases as attributes.\n"
            "• Dependency (presentation/dependency): Builders/factories exposing repositories ONLY for DI. metadata.providers must list provider functions returning interfaces.\n\n"
            "Example Router method:\n"
            "def create_order_endpoint(request: CreateOrderRequest, order_repo: IOrderRepository):\n"
            "    use_case = CreateOrderUseCase(order_repo)\n"
            "    return use_case.execute(request)\n\n"
            "ROUTER METHOD DESCRIPTION TEMPLATE (MANDATORY):\n"
            "Each Router method's 'description' MUST follow exactly this 3-line template so codegen can parse it:\n"
            "[HTTP] <METHOD> <PATH>\n"
            "Response: <ResponseDTO> (status=<CODE>)\n"
            "Behavior: Instantiate <UseCaseClass>(deps...) and execute(request)\n\n"
            "Generate only classes needed for this specific User Flow. Ensure all schema fields (properties, attributes, metadata) are populated meaningfully.\n"
        ),
    )


def verify_components_agent(model_name: str = "openai:o4-mini"):
    """
    Create an agent to verify classes.

    Args:
        model_name: The name of the LLM model to use

    Returns:
        Agent configured to verify classes
    """
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="low",
        openai_reasoning_summary="detailed",
        temperature=0.2,
    )
    return Agent(
        model_name,
        output_type=List[ArchitectureComponent],
        retries=3,
        model_settings=settings,
        system_prompt=(
            "You are a Clean Architecture validator. Verify and fix class designs to strictly follow Onion Architecture principles and the JSON schema.\n\n"
            "VALIDATION CHECKLIST:\n"
            "✓ Use Cases have exactly ONE public method named 'execute'\n"
            "✓ Domain entities contain NO I/O operations (save, fetch, load, etc.)\n"
            "✓ Application layer depends ONLY on interfaces (I-prefixed)\n"
            "✓ Infrastructure classes implement their corresponding interfaces\n"
            "✓ Correct layer tags and naming conventions (PascalCase classes, snake_case methods)\n"
            "✓ Every class must satisfy schema fields: dependencies, properties/attributes, methods, metadata. Fill missing metadata with structured details.\n"
            "✓ Repository naming: I<Entity>Repository → <Tech><Entity>Repository\n"
            "✓ Adapter naming: I<Capability>Adapter → <Tech><Capability>Adapter\n\n"
            "FIXES TO APPLY:\n"
            "• Consolidate duplicate classes: merge attributes/methods from duplicates\n"
            "• Ensure dependencies list all injected interfaces/adapters\n"
            "• Populate properties or metadata.fields for domain entities and DTO schemas.\n"
            "• metadata.contract for interfaces must document each method signature with inputs/outputs.\n"
            "• Remove cross-cutting concerns (auth, logging, HTTP handling)\n"
            "• Fix incorrect layer assignments\n"
            "• PRESERVE all Use Cases\n"
            "• Ensure Presentation Routers follow description template and metadata includes route + request/response DTOs.\n"
            "• Dependency providers must expose get_*_repository methods returning interfaces; metadata.providers must list them.\n"
            "• ENHANCE Use Case metadata.inputs/outputs/dependencies/steps; ensure methods describe orchestration.\n"
            "• Infrastructure adapters/repositories must include metadata.endpoint/method/payload_schema or metadata.storage details (tech + model).\n\n"
            "DEPENDENCY RULE ENFORCEMENT:\n"
            "Presentation depends on Application; Application depends on Domain; Infrastructure implements Application interfaces.\n"
            "Inner layers NEVER depend on outer layers (inward dependency rule).\n\n"
            "Return ONLY the corrected BEObject list matching the JSON schema."
        ),
    )


# ---------------------------------------------------------------------------
# Parser pipeline
# ---------------------------------------------------------------------------


class BEClassParser:
    """
    BEClassParser module: parse PRD to classes using Pydantic AI.
    """

    def __init__(self, model_name: str = "openai:o4-mini"):
        self.model_name = model_name
        self.prd_to_userflow = prd_to_flow_agent(model_name)
        self.userflow_to_class = flow_to_component_agent(model_name)
        self.verify_classes = verify_components_agent(model_name)

    async def process(self, prd_text: str) -> str:
        """
        Extract necessary classes that implement PRD requirements.

        Args:
            prd_text: The Product Requirements Document text

        Returns:
            List of dictionaries containing class definitions for each feature

        Raises:
            Exception: If any stage of the pipeline fails
        """
        user_flows_output: UserFlowOutput = (
            await self.prd_to_userflow.run(prd_text)
        ).output
        user_flows = user_flows_output.userflows
        project_description = user_flows_output.project_description
        logger.info("Generated %s user flows", len(user_flows))

        feature_components: List[Dict[str, Any]] = []
        for uf in user_flows:
            logger.info("Generating classes for feature: %s", uf.feature)
            userflow_payload = json.dumps(
                {
                    "project_description": project_description,
                    "user_flow": uf.model_dump(),
                },
                ensure_ascii=False,
            )
            components = (await self.userflow_to_class.run(userflow_payload)).output
            components = [component.model_dump() for component in components]
            feature_components.append(
                {
                    "feature": uf.feature,
                    "user_flow": uf.model_dump(),
                    "classes": components,
                }
            )
        logger.info("Verifying classes")
        verified_components = (
            await self.verify_classes.run(
                json.dumps(feature_components, ensure_ascii=False)
            )
        ).output
        verified_components = [
            component.model_dump() for component in verified_components
        ]

        validated_components: List[Dict[str, Any]] = []
        for component in verified_components:
            component_with_defaults = _apply_component_defaults(component)
            try:
                component_model = ArchitectureComponent.model_validate(
                    component_with_defaults
                )
                validated_components.append(component_model.model_dump())
            except ValidationError as exc:
                logger.warning(
                    "Class validation failed for %s: %s",
                    component_with_defaults.get("name"),
                    exc,
                )
                validated_components.append(component_with_defaults)

        return json.dumps(validated_components, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def main():
    """Main function to run the class parser pipeline."""
    prd_path = "projects/warehouse/PRD.md"

    if not os.path.exists(prd_path):
        logger.error("PRD file not found: %s", prd_path)
        sys.exit(1)

    with open(prd_path, "r", encoding="utf-8") as f:
        prd_text = f.read()

    parser = BEClassParser()
    print("\n=== Full Pipeline Output ===")
    verified_classes_json = await parser.process(prd_text)
    verified_classes = json.loads(verified_classes_json)
    with open("projects/warehouse/be_objects.json", "w", encoding="utf-8") as f:
        json.dump(verified_classes, f, ensure_ascii=False, indent=2)
    print("Final output saved to projects/warehouse/be_objects.json")


# verify dựa trên dữ liệu trung gian đã có
# async def main_verify_classes():
#     class_verifier = verify_classes_agent()
#     print("\n=== Full Pipeline Output ===")
#     feature_classes = json.load(
#         open("projects/warehouse/feature_classes.json", "r", encoding="utf-8")
#     )
#     verified_classes = (await class_verifier.run(str(feature_classes))).output
#     verified_classes = [class_item.model_dump() for class_item in verified_classes]
#     with open("projects/warehouse/be_objects.json", "w", encoding="utf-8") as f:
#         json.dump(verified_classes, f, ensure_ascii=False, indent=2)
#     print("Final output saved to projects/warehouse/be_objects.json")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    # asyncio.run(main_verify_classes())
