"""
Feature Suggestion Module

Provides functionality to suggest features for software projects by user role and priority using LLMs.
"""

import logging
from typing import List, Literal
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# Setup logger
logger = logging.getLogger(__name__)

# System prompt for the LLM agent
SYSTEM_PROMPT = (
    "Bạn là một AI gợi ý tính năng phần mềm. Dựa vào mô tả dự án đầu vào, hãy liệt kê chi tiết các tính năng "
    "theo từng vai trò người dùng, và đánh dấu mức độ ưu tiên 'must-have', 'nice-to-have', hoặc 'optional'."
)


class FeatureItem(BaseModel):
    """A feature item with name, priority, and optional description."""

    name: str = Field(..., description="Feature name")
    priority: Literal["must-have", "nice-to-have", "optional"] = Field(
        ..., description="Feature priority"
    )
    description: str = Field(None, description="Feature description")


class RoleFeatures(BaseModel):
    """Features grouped by user role."""

    role: str = Field(..., description="User role")
    features: List[FeatureItem] = Field(
        ..., description="List of features for this role"
    )


class FeatureSuggestion(BaseModel):
    """Feature suggestions for all roles."""

    suggestions: List[RoleFeatures] = Field(
        ..., description="List of feature suggestions by role"
    )


class FeatureSuggestionAgent:
    """Agent for suggesting features for a project by user role and priority."""

    def __init__(self, model: str = "openai:gpt-4o"):
        """Initialize the FeatureSuggestionAgent.

        Args:
            model: The AI model to use for feature suggestion (default: "openai:gpt-4o")
        """
        self.model = model
        self.agent = Agent(
            model,
            system_prompt=SYSTEM_PROMPT,
            output_type=FeatureSuggestion,
        )
        logger.info(f"Initialized FeatureSuggestionAgent with model: {model}")

    async def suggest(self, input_text: str) -> FeatureSuggestion:
        """Suggest features for the given project description.

        Args:
            input_text: The project description
        Returns:
            FeatureSuggestion: Suggested features grouped by role and priority
        """
        if not input_text or not input_text.strip():
            raise ValueError("Input text for feature suggestion cannot be empty.")
        result = await self.agent.run(input_text)
        return result.output


# # Example usage
# if __name__ == "__main__":
#     import asyncio

#     logging.basicConfig(level=logging.INFO)
#     INPUT_TEXT = (
#         "Hệ thống quản lý kho tự động: theo dõi tồn kho, xử lý đơn hàng, "
#         "báo cáo doanh số. Phân quyền người dùng cho nhân viên kho và quản lý."
#     )
#     agent = FeatureSuggestionAgent()
#     result = asyncio.run(agent.suggest(INPUT_TEXT))
#     logger.info(result.model_dump_json(indent=2))
