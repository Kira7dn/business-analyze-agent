"""
User Story Parser: LLM pipeline to convert PRD -> User Flow -> Data Flow -> Function Spec

This module uses PydanticAI to define agents that process a Product Requirements Document
through multiple transformation stages to generate function specifications.

The pipeline follows these steps:
1. Convert PRD text to UserFlow objects
2. Convert UserFlow to DataFlow
3. Convert DataFlow to FunctionSpec
"""

from typing import List, Optional
import json
import logging
import sys
import os
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

# from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


# 1. Define schema for each step


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

    class Config:
        """Configuration for the UserFlow model."""

        json_schema_extra = {
            "example": {
                "feature": "User Registration",
                "actor": "New Customer",
                "steps": [
                    "User navigates to registration page",
                    "User fills out registration form",
                    "User submits the form",
                    "System validates input",
                    "System creates user account",
                ],
            }
        }


class UserFlowOutput(BaseModel):
    """Collection of user flows extracted from PRD."""

    userflows: List[UserFlow] = Field(
        description="List of all user flows extracted from the PRD."
    )
    project_description: str = Field(description="Project description.")

    class Config:
        """Configuration for the UserFlowList model."""

        json_schema_extra = {
            "example": {
                "userflows": [
                    {
                        "feature": "Dashboard summary",
                        "actor": "Warehouse staff",
                        "steps": ["Access dashboard", "Filter orders", "View results"],
                    },
                    {
                        "feature": "Confirm order",
                        "actor": "Warehouse staff",
                        "steps": [
                            "Select order",
                            "Confirm order",
                            "Update status",
                        ],
                    },
                ],
                "project_description": "Build a web application for order management that helps warehouse staff process at least 1,000 orders per day with an average response time of ≤ 150ms.",
            }
        }


class DataFlow(BaseModel):
    """Data flow for a feature."""

    project_description: str = Field(description="Project description.")
    feature: str = Field(description="Name of the feature (e.g., 'OrderConfirmation').")
    actor: str = Field(description="Name of the actor (e.g., 'Warehouse staff').")
    description: str = Field(
        description="Overall description of the data flow in 2-3 sentences."
    )
    data_flow_steps: List[str] = Field(
        default_factory=list,
        description="List of backend data operations in VerbNoun format (e.g., 'ValidateOrder', 'SaveCustomer'). Each step maps to a method call in the Use Case execute() method.",
    )

    class Config:
        """Configuration for the DataFlow model."""

        json_schema_extra = {
            "example": {
                "project_description": "Build a web application for order management that helps warehouse staff process at least 1,000 orders per day with an average response time of ≤ 150ms.",
                "feature": "OrderConfirmation",
                "actor": "Warehouse staff",
                "description": "Handles the process of collecting, validating, and storing new user registration data",
                "data_flow_steps": [
                    "ValidateUserInput: Validates that user input meets requirements",
                    "StoreUserInDatabase: Stores user data in the database",
                ],
            }
        }


class MethodSpec(BaseModel):
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

    class Config:
        """Configuration for the FunctionSpec model."""

        json_schema_extra = {
            "example": {
                "method_name": "search_products",
                "description": "Searches for products based on keywords and filters",
                "parameters": ["keyword: str", "page: int = 1"],
                "return_type": "dict",
            }
        }


class ClassDefinition(BaseModel):
    """Class definition for a feature."""

    class_name: str = Field(description="Name of the class in PascalCase format.")
    layer: str = Field(
        description=(
            "Deterministic architectural layer tag of the class. One of: "
            "'domain/entity', 'domain/service', "
            "'application/interface', 'application/use_case', "
            "'infrastructure/model', 'infrastructure/repository', 'infrastructure/adapter', "
            "'presentation/schema', 'presentation/dependency', 'presentation/router'."
        )
    )
    dependencies: Optional[List[str]] = Field(
        default_factory=list,
        description=(
            "List of interface names this class depends on (e.g., ['IOrderRepository', 'IPaymentAdapter'])."
        ),
    )
    description: str = Field(
        description="Description of the class's purpose and role within the architecture."
    )
    attributes: Optional[List[str]] = Field(
        default_factory=list,
        description="List of attributes in format 'name: type'.",
    )
    methods: Optional[List[MethodSpec]] = Field(
        default_factory=list,
        description="List of methods in MethodSpec format.",
    )

    class Config:
        """Configuration for the ClassDefinition model."""

        json_schema_extra = {
            "example": {
                "class_name": "SQLOrderRepository",
                "layer": "infrastructure/repository",
                "description": "Implements IOrderRepository using SQLAlchemy with PostgreSQL.",
                "attributes": ["db: Session"],
                "dependencies": ["IOrderRepository"],
                "methods": [
                    {
                        "method_name": "save",
                        "description": "Persist an order and return the stored entity with assigned id.",
                        "parameters": ["order: Order"],
                        "return_type": "Order",
                    }
                ],
            }
        }


# 2. Define agent factories for each step


def prd_to_userflow_agent(model_name: str = "openai:o4-mini"):
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


def userflow_to_dataflow_agent(model_name: str = "openai:o4-mini"):
    """
    Create an agent to convert UserFlow to DataFlow.

    Args:
        model_name: The name of the LLM model to use

    Returns:
        Agent configured to convert UserFlow to DataFlow
    """
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="low",
        openai_reasoning_summary="detailed",
        temperature=0.2,
    )
    return Agent(
        model_name,
        output_type=DataFlow,
        model_settings=settings,
        retries=3,
        system_prompt=(
            "You are a backend architect specializing in Clean Architecture DataFlow design. Convert User Flow to DataFlow that maps 1:1 to a Use Case with single 'execute()' method.\n\n"
            "DATAFLOW = USE CASE MAPPING:\n"
            "• DataFlow input parameters → Use Case execute() parameters\n"
            "• DataFlow output → Use Case execute() return type\n"
            "• DataFlow steps → orchestration logic inside execute()\n\n"
            "STEP NAMING RULES:\n"
            "• Use VerbNoun format (e.g., 'ValidateOrder', 'SaveCustomer', 'SendEmail')\n"
            "• Each step maps to either:\n"
            "  - Domain entity method (intrinsic behavior)\n"
            "  - Repository interface method (persistence)\n"
            "  - Adapter interface method (external service)\n"
            "• NO UI, auth, logging, DI, or cross-cutting concerns\n"
            "• Steps must be backend data operations only\n\n"
            "CONSTRAINTS:\n"
            "• Include ONLY steps necessary for the User Flow\n"
            "• No optimization or 'nice-to-have' steps\n"
            "• Each step should be implementable as a single method call\n"
            "• Steps should follow business logic sequence\n\n"
            "OUTPUT FORMAT:\n"
            "- project_description: from input\n"
            "- feature: PascalCase feature name\n"
            "- actor: from User Flow\n"
            "- description: 2-3 sentences explaining the DataFlow purpose\n"
            "- data_flow_steps: minimal list of VerbNoun operations\n"
        ),
    )


def dataflow_to_class_agent(model_name: str = "openai:o4-mini"):
    """
    Create an agent to convert DataFlow to ClassOutput.

    Args:
        model_name: The name of the LLM model to use

    Returns:
        Agent configured to convert DataFlow to ClassOutput
    """
    settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort="low",
        openai_reasoning_summary="detailed",
        temperature=0.1,
    )
    return Agent(
        model_name,
        output_type=List[ClassDefinition],
        retries=3,
        model_settings=settings,
        system_prompt=(
            "You are a backend architect specializing in Onion/Clean Architecture. Design minimal OOP classes from the DataFlow.\n\n"
            "CRITICAL RULES:\n"
            "1. Inner layers must NOT depend on outer layers (inward dependency rule).\n"
            "2. Presentation depends on Application; Application depends on Domain; Infrastructure implements Application interfaces.\n"
            "3. Domain entities have NO I/O operations (no save, fetch, etc.).\n"
            "4. Application layer depends ONLY on interfaces (ports), never implementations.\n"
            "5. Infrastructure implements Application interfaces.\n\n"
            "FORMAT:\n"
            "- class_name: PascalCase\n"
            "- attributes/methods: snake_case\n"
            "- dependencies: list of interface names this class depends on (e.g., ['IOrderRepository', 'IPaymentAdapter'])\n"
            "- layer: exact tags from: 'domain/entity', 'domain/service', 'application/interface', 'application/use_case', 'infrastructure/model', 'infrastructure/repository', 'infrastructure/adapter', 'presentation/schema', 'presentation/router', 'presentation/dependency'\n\n"
            "LAYER SPECIFICATIONS:\n"
            "Domain Layer:\n"
            "• Entities (domain/entity): Business objects with intrinsic behaviors only. NO persistence methods.\n"
            "• Services (domain/service): Pure business algorithms. May depend on other domain services.\n"
            "Application Layer:\n"
            "• Use Case (application/use_case): Orchestrates DataFlow steps via interfaces. Single 'execute()' method maps 1:1 to DataFlow input/output. Dependencies are injected interfaces.\n"
            "  IMPORTANT: Method descriptions must include detailed step-by-step execution logic:\n"
            "  - Validation steps (what to validate and how)\n"
            "  - Domain method calls (which entity/service methods to call)\n"
            "  - Repository interactions (fetch, save operations)\n"
            "  - Business logic sequence (order of operations)\n"
            "  - Error handling scenarios\n"
            "  Example: 'Validates request.items not empty, creates Order entity via Order.create(), validates business rules via order.validate_new(), saves via repository.save(), logs audit via audit_repository.log(), returns CreateOrderResponse with order.id'\n"
            "• Interface (application/interface): Abstract ports for repositories/adapters. No attributes.\n"
            "Infrastructure Layer:\n"
            "• Repository (infrastructure/repository): Implements application interfaces.\n"
            "• Adapter (infrastructure/adapter): Implements application interfaces.\n"
            "• Model (infrastructure/model): ORM/database mapping classes. Mirror Domain entities structure but focused on persistence. No business logic.\n"
            ""
            "Presentation Layer:\n"
            "• Schema (presentation/schema): Request/Response DTOs. No methods.\n"
            "• Router (presentation/router): HTTP endpoint handlers. Each method accepts Request DTO + repository providers as parameters, instantiates Use Case internally, calls execute(request), returns Response DTO. Do NOT store Use Cases as attributes.\n"
            "• Dependency (presentation/dependency): Builders/factories exposing repositories ONLY for DI. Expose provider-style functions for repositories with interface return types (e.g., get_*_repository() -> I*Repository).\n\n"
            "Example Router method:\n"
            "def create_order_endpoint(request: CreateOrderRequest, order_repo: IOrderRepository):\n"
            "    use_case = CreateOrderUseCase(order_repo)\n"
            "    return use_case.execute(request)\n\n"
            "ROUTER METHOD DESCRIPTION TEMPLATE (MANDATORY):\n"
            "Each Router method's 'description' MUST follow exactly this 3-line template so codegen can parse it:\n"
            "[HTTP] <METHOD> <PATH>\n"
            "Response: <ResponseDTO> (status=<CODE>)\n"
            "Behavior: Instantiate <UseCaseClass>(deps...) and execute(request)\n\n"
            "Generate only classes needed for this specific DataFlow.\n"
        ),
    )


def verify_classes_agent(model_name: str = "openai:o4-mini"):
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
        output_type=List[ClassDefinition],
        retries=3,
        model_settings=settings,
        system_prompt=(
            "You are a Clean Architecture validator. Verify and fix class designs to strictly follow Onion Architecture principles from the guide.\n\n"
            "VALIDATION CHECKLIST:\n"
            "✓ Use Cases have exactly ONE public method named 'execute'\n"
            "✓ Domain entities contain NO I/O operations (save, fetch, load, etc.)\n"
            "✓ Application layer depends ONLY on interfaces (I-prefixed)\n"
            "✓ Infrastructure classes implement their corresponding interfaces\n"
            "✓ Correct layer tags and naming conventions (PascalCase classes, snake_case methods)\n"
            "✓ Layers must use: application/interface, application/use_case, infrastructure/model, infrastructure/repository, infrastructure/adapter, presentation/schema, presentation/dependency, presentation/router\n"
            "✓ Repository naming: I<Entity>Repository → <Tech><Entity>Repository\n"
            "✓ Adapter naming: I<Capability>Adapter → <Tech><Capability>Adapter\n\n"
            "FIXES TO APPLY:\n"
            "• Consolidate duplicate classes: merge attributes/methods from all instances of same class\n"
            "• Ensure 'dependencies' lists interface names required by the class; add if missing\n"
            "• Remove cross-cutting concerns (auth, logging, HTTP handling)\n"
            "• Fix incorrect layer assignments\n"
            "• PRESERVE all Use Cases - each feature needs its own Use Case\n"
            "• Create unified domain model that supports all features\n"
            "• Ensure Presentation Routers: endpoint methods accept repository providers as parameters and instantiate Use Case internally\n"
            "• Dependency providers must expose get_*_repository methods ONLY (no get_*_use_case) and return interface types (e.g., get_product_repo(...) -> IProductRepository)\n"
            "• Enforce Router method description template:\n"
            "  [HTTP] <METHOD> <PATH>\n"
            "  Response: <ResponseDTO> (status=<CODE>)\n"
            "  Behavior: Instantiate <UseCaseClass>(deps...) and execute(request)\n"
            "  Example: def create_order_endpoint(request: CreateOrderRequest, order_repo: IOrderRepository):\n"
            "• ENHANCE Use Case descriptions with detailed execution logic (validation, domain calls, repository interactions, business rules, error handling)\n"
            "  Example: 'Validates request.order_id exists, fetches order via repository.get_by_id(), checks order.status == \"pending\", calls order.confirm(), saves via repository.update(), returns ConfirmOrderResponse'\n\n"
            "DEPENDENCY RULE ENFORCEMENT:\n"
            "Presentation depends on Application; Application depends on Domain; Infrastructure implements Application interfaces.\n"
            "Inner layers NEVER depend on outer layers (inward dependency rule).\n\n"
            "Return ONLY the corrected ClassDefinition list."
        ),
    )


# 3. Main pipeline function


class ClassParser:
    """
    ClassParser module: parse PRD to classes using Pydantic AI.
    """

    def __init__(self, model_name: str = "openai:o4-mini"):
        self.model_name = model_name
        self.prd_to_userflow = prd_to_userflow_agent(model_name)
        self.userflow_to_dataflow = userflow_to_dataflow_agent(model_name)
        self.dataflow_to_class = dataflow_to_class_agent(model_name)
        self.verify_classes = verify_classes_agent(model_name)

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
        logger.info(f"Generated {len(user_flows)} user flows")
        print(f"Generated {len(user_flows)} user flows")
        with open("projects/warehouse/user_flows.json", "w", encoding="utf-8") as f:
            json.dump([uf.model_dump() for uf in user_flows], f)
        feature_classes = []
        for uf in user_flows:
            logger.info(f"Converting userflow to dataflow for feature: {uf.feature}")
            print(f"Converting userflow to dataflow for feature: {uf.feature}")
            uf_info = f"Project Description: {project_description}\nUser Flow: {uf.model_dump()}"
            data_flow: DataFlow = (
                await self.userflow_to_dataflow.run(uf_info)
            ).output.model_dump()
            with open(
                f"projects/warehouse/data_flows/{uf.feature}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(data_flow, f)
            logger.info(f"Converting dataflow to classes for feature: {uf.feature}")
            print(f"Converting dataflow to classes for feature: {uf.feature}")
            classes = (await self.dataflow_to_class.run(str(data_flow))).output
            # convert classes to json
            classes = [class_item.model_dump() for class_item in classes]
            logger.info(f"Generated classes for feature: {uf.feature}")
            print(f"Generated classes for feature: {uf.feature}")
            feature_classes.append(
                {
                    "feature": uf.feature,
                    "data_flow": data_flow,
                    "classes": classes,
                }
            )
        with open(
            "projects/warehouse/feature_classes.json", "w", encoding="utf-8"
        ) as f:
            json.dump(feature_classes, f)
        logger.info("Verifying classes")
        print("Verifying classes")
        verified_classes = (await self.verify_classes.run(str(feature_classes))).output
        # convert verified_classes to json
        verified_classes = [class_item.model_dump() for class_item in verified_classes]
        with open(
            "projects/warehouse/verified_classes.json", "w", encoding="utf-8"
        ) as f:
            json.dump(verified_classes, f)
        verified_classes = json.dumps(verified_classes)
        return verified_classes


# Example usage
async def main():
    """Main function to run the class parser pipeline."""
    prd_path = "projects/warehouse/PRD.md"

    if not os.path.exists(prd_path):
        logger.error(f"PRD file not found: {prd_path}")
        sys.exit(1)

    with open(prd_path, "r", encoding="utf-8") as f:
        prd_text = f.read()

    parser = ClassParser()
    print("\n=== Full Pipeline Output ===")
    await parser.process(prd_text)
    # with open("projects/warehouse/classes.json", "w", encoding="utf-8") as f:
    #     json.dump(output, f, ensure_ascii=False, indent=4)
    print("Final output saved to projects/warehouse/classes.json")


async def main_verify_classes():
    class_verifier = verify_classes_agent()
    print("\n=== Full Pipeline Output ===")
    feature_classes = json.load(
        open("projects/warehouse/feature_classes.json", "r", encoding="utf-8")
    )
    verified_classes = (await class_verifier.run(str(feature_classes))).output
    verified_classes = [class_item.model_dump() for class_item in verified_classes]
    with open("projects/warehouse/verified_classes.json", "w", encoding="utf-8") as f:
        json.dump(verified_classes, f, ensure_ascii=False, indent=4)
    print("Final output saved to projects/warehouse/verified_classes.json")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main_verify_classes())
