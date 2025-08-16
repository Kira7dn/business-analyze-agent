"""
User Story Parser: LLM pipeline to convert PRD -> User Flow -> Data Flow -> Function Spec

This module uses PydanticAI to define agents that process a Product Requirements Document
through multiple transformation stages to generate function specifications.

The pipeline follows these steps:
1. Convert PRD text to UserFlow objects
2. Convert UserFlow to DataFlow
3. Convert DataFlow to FunctionSpec
"""

from typing import Dict, List
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
        description="Name of the feature (e.g., 'User Registration', 'Product Search')."
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
        description="List of all data flow steps in string format.",
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

    class_name: str = Field(description="Name of the class in snake_case format.")
    layer: str = Field(
        description="The architectural layer/role of the class (e.g., 'Entity', 'Service', 'Repository Interface', 'Repository Implementation')."
    )
    description: str = Field(
        description="Description of the class's purpose and role within the architecture."
    )
    attributes: List[str] = Field(
        default_factory=list,
        description="List of attributes in format 'name: type'.",
    )
    methods: List[MethodSpec] = Field(
        default_factory=list,
        description="List of methods in MethodSpec format.",
    )

    class Config:
        """Configuration for the ClassDefinition model."""

        json_schema_extra = {
            "example": {
                "class_name": "User",
                "layer": "Data Model",
                "description": "Data model for a user in the system.",
                "attributes": ["email: string", "age: integer"],
                "methods": [
                    {
                        "method_name": "search_products",
                        "description": "Searches for products based on keywords and filters",
                        "parameters": ["keyword: str", "page: int = 1"],
                        "return_type": "dict",
                    }
                ],
            }
        }


class ClassOutput(BaseModel):
    """Collection of class definitions for implementing PRD requirements."""

    classes: List[ClassDefinition] = Field(
        description="List of all necessary classes to implement the PRD requirements."
    )


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
            "You are a requirements analyst. Carefully read the following PRD (Product Requirements Document) and extract the following information as a single object:\n\n"
            "1.  All distinct 'Feature Suggestions by User Role'. For each feature, create a UserFlow object with the following details:\n"
            "* `feature`: The name of the feature.\n"
            "* `actor`: The user role associated with the feature.\n"
            "* `steps`: A list of necessary sequential actions or steps within that user flow, additional action or step is not allowed.\n"
            "2. The overall 'Project description'\n"
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
            "You are an expert Software Architect and System Designer specializing in backend data flows. Your task is to provided **Project Description** and **User Flow** to design a **Data Flow** that outlines the necessary backend data operations. Only necessary steps, additional steps or optimization steps are not allowed.\n"
            "\n"
            "For each data flow, use the following structure:\n"
            "- **project_description**: The project description (e.g., 'Build a web application for order management that helps warehouse staff process orders.')\n"
            "- **feature**: The name of the feature (e.g., 'OrderConfirmation')\n"
            "- **actor**: The name of the user or system interacting with the feature (e.g., 'Warehouse staff')\n"
            "- **description**: A concise summary of the data operations required for this feature.\n"
            "- **data_flow_steps**: A minimal list detailing the data transformation and manipulation steps performed in the backend to meet the feature requirements outlined in the user flow. Only necessary steps, additional steps or optimization steps are not allowed (e.g., ['FetchUnconfirmedOrders: Retrieves all unconfirmed orders from the database']).\n"
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
        temperature=0.2,
    )
    return Agent(
        model_name,
        output_type=List[ClassDefinition],
        retries=3,
        model_settings=settings,
        system_prompt=(
            "You are a backend architect. Based on the **DataFlow**, design **minimal OOP classes** following a **Onion Architecture pattern**. Focus solely on **Data Models, Repositories, and Services**.\n\n"
            "For each feature must include these layers:\n"
            "1.  **Entity:** Core business entity. Attributes: 'name: type'. Methods: Intrinsic behaviors only (no CRUD).\n"
            "2.  **Service(Use Case):** Business logic layer. Orchestrates operations using Repositories. Attributes: Dependencies (e.g., 'order_repo: OrderRepository'). Methods: Business actions (e.g., 'create_order', 'confirm_order'). Do NOT interact directly with data store.\n"
            "3.  **Repository:** Defines data access operations.\n"
            "    - **Interface (Port):** Abstract definition of data operations (e.g., 'OrderRepository'). Located in the **Application layer**. Attributes: None. Methods: Abstract CRUD/query methods.\n"
            "    - **Implementation (Adapter):** Concrete implementation of the Repository Interface (e.g., 'SQLOrderRepository'). Located in the **Infrastructure layer**. Attributes: Database connection/ORM. Methods: Implements interface methods, interacts with data store.\n"
            "**Guidelines:**\n"
            "- Each class should include class_name: string, layer: string, description: string, attributes: list, methods: list"
            "- Use **PascalCase** for class names, **snake_case** for attributes/methods.\n"
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
            "You are a senior software architecture assistant specializing in **Onion Architecture** for enterprise systems. "
            "Your task is to analyze and optimize the provided class designs, ensuring they strictly adhere to the Onion Architecture pattern.\n\n"
            "The input JSON contains:\n"
            "- A list of classes with their definitions (name, layer, description, attributes, methods).\n"
            "- These classes might mix business logic with auxiliary concerns or inappropriate architectural roles.\n\n"
            "## 🎯 GOAL:\n"
            "Redesign the class model to a **minimal, clean, and optimized Object-Oriented structure** that strictly follows the Onion Architecture pattern, focusing ONLY on **Data Models, Repositories, and Services**.\n\n"
            "## ❌ REMOVE:\n"
            "- Any class or method related to **technical infrastructure or cross-cutting concerns**, including:\n"
            "  - Authentication, authorization, OAuth2, session or token validation\n"
            "  - Logging, error handling/wrapping, response formatting\n"
            "  - Database connection management (outside of what a Repository needs to abstract)\n"
            "- Service or orchestration methods that do not directly contribute to the core business flow (e.g., purely technical orchestration).\n\n"
            "## ✅ KEEP & OPTIMIZE:\n"
            "- Retain and refine only **essential Entities**, **Services**, **Repository Interfaces**, and **Repository Implementations** directly tied to the core business logic of the DataFlow.\n"
            "- Ensure each class has a **single, well-defined responsibility** according to its Onion Architecture role:\n"
            "  - **Entity:** Represent core business entities; contain attributes and intrinsic behaviors only (no CRUD methods).\n"
            "  - **Service:** Encapsulate business rules and workflows; orchestrate operations using Repositories; do NOT directly interact with the data store.\n"
            "  - **Repository Interface:** Abstract definition of data operations; located in the **Application layer**. Attributes: None. Methods: Abstract CRUD/query methods.\n"
            "  - **Repository Implementation:** Concrete implementation of the Repository Interface; located in the **Infrastructure layer**. Attributes: Database connection/ORM. Methods: Implements interface methods, interacts with data store.\n"
            "  - **Repository Interface:** Abstract definition of data operations; located in the **Application layer**. Attributes: None. Methods: Abstract CRUD/query methods.\n"
            "  - **Repository Implementation:** Concrete implementation of the Repository Interface; located in the **Infrastructure layer**. Attributes: Database connection/ORM. Methods: Implements interface methods, interacts with data store.\n"
            "- Eliminate redundancy by consolidating equivalent entities or logic.\n"
            "- Rename classes (PascalCase) and methods/attributes (snake_case) to reflect clear business actions and architectural roles.\n\n"
            "## 📤 OUTPUT:\n"
            "Return the following information as a single object:\n"
            "- classes: List of optimized classes (Data Models, Repositories, Services) in the correct format.\n"
            "- notes: Explain how these optimized classes, designed under the Onion Architecture pattern, collaborate to fulfill the feature’s business logic and requirements."
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

    async def process(self, prd_text: str) -> List[Dict]:
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
        feature_classes = []
        for uf in user_flows:
            logger.info(f"Converting userflow to dataflow for feature: {uf.feature}")
            print(f"Converting userflow to dataflow for feature: {uf.feature}")
            uf_info = f"Project Description: {project_description}\nUser Flow: {uf.model_dump()}"
            data_flow = (
                await self.userflow_to_dataflow.run(uf_info)
            ).output.model_dump()
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
        logger.info("Verifying classes")
        print("Verifying classes")
        verified_classes = (await self.verify_classes.run(str(feature_classes))).output
        # convert verified_classes to json
        verified_classes = [class_item.model_dump() for class_item in verified_classes]
        return verified_classes


# Example usage
async def main():
    """Main function to run the class parser pipeline."""
    prd_path = "projects/warehouse/prd_text.md"

    if not os.path.exists(prd_path):
        logger.error(f"PRD file not found: {prd_path}")
        sys.exit(1)

    with open(prd_path, "r", encoding="utf-8") as f:
        prd_text = f.read()

    parser = ClassParser()
    print("\n=== Full Pipeline Output ===")
    output = await parser.process(prd_text)
    with open("projects/warehouse/classes.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print("Final output saved to projects/warehouse/classes.json")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
