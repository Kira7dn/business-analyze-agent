"""
Requirement Evaluation Module (Version 2)

This module provides functionality to evaluate project requirements using AI-based analysis.
It uses the pydantic-ai library to analyze requirements against predefined criteria
and generates evaluation reports with scores and improvement questions.
"""

import sys
import io
import logging
from typing import Dict, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# Setup logger
logger = logging.getLogger(__name__)


# Evaluation output schema
class Question(BaseModel):
    """Question model for evaluation feedback.

    Attributes:
        id: Identifier for the criterion this question relates to
        question: The actual question text for clarification
    """

    id: str = Field(..., description="Criterion ID this question relates to")
    question: str = Field(..., description="Question text for clarification")


class EvaluationResult(BaseModel):
    """Evaluation result model containing scores and improvement questions.

    Attributes:
        scores: Dictionary mapping criterion IDs to their scores (0-5)
        questions: List of questions for criteria that need clarification
        overall_score: Average score across all criteria
    """

    scores: Dict[str, int] = Field(
        ...,
        description="Dictionary of criterion IDs to scores (0-5)",
        example={"project_goal": 4, "target_users": 3},
    )
    questions: List[Question] = Field(
        default_factory=list,
        description="List of questions for criteria that need clarification",
    )
    overall_score: float = Field(
        ..., ge=0, le=5, description="Average score across all criteria (0-5)"
    )


# MCP evaluation schema
EVALUATION_CRITERIA = [
    {
        "id": "project_goal",
        "name": "Mục tiêu dự án",
        "description": "Mô tả rõ mục tiêu tổng quát của dự án, tại sao dự án này quan trọng và cái gì cần đạt được.",
        "score_0": "Không đề cập mục tiêu hoặc mục tiêu rất mơ hồ, không rõ ràng.",
        "score_5": "Mục tiêu được định nghĩa rất rõ ràng, chi tiết, có tính cụ thể và phù hợp với bối cảnh dự án.",
        "question": "Mục tiêu cụ thể của dự án này là gì? Tại sao nó quan trọng?",
    },
    {
        "id": "target_users",
        "name": "Đối tượng người dùng",
        "description": "Xác định nhóm người dùng hoặc stakeholder chính của hệ thống.",
        "score_0": "Không xác định rõ đối tượng người dùng hoặc hoàn toàn thiếu thông tin.",
        "score_5": "Đối tượng người dùng được mô tả rõ ràng (độ tuổi, nghề nghiệp, nhu cầu, ...) phù hợp với mục tiêu dự án.",
        "question": "Ai là người dùng chính của hệ thống này?",
    },
    {
        "id": "core_features",
        "name": "Tính năng cốt lõi",
        "description": "Liệt kê các chức năng chính quan trọng mà hệ thống cần có để đạt mục tiêu.",
        "score_0": "Không đề cập hoặc chỉ nêu quá ít tính năng, thiếu các chức năng cơ bản.",
        "score_5": "Các tính năng chính được liệt kê đầy đủ, chi tiết và rõ ràng, bao gồm cả chức năng phụ trợ cần thiết.",
        "question": "Các tính năng chính nào cần triển khai để đáp ứng mục tiêu dự án?",
    },
    {
        "id": "nonfunctional_constraints",
        "name": "Ràng buộc phi chức năng",
        "description": "Các yêu cầu phi chức năng như hiệu năng, bảo mật, khả năng mở rộng, độ ổn định...",
        "score_0": "Không có bất kỳ thông tin nào về các yêu cầu phi chức năng (ví dụ: hiệu suất, bảo mật, ...).",
        "score_5": "Đề cập đầy đủ và cụ thể các ràng buộc phi chức năng quan trọng (như thời gian phản hồi, tiêu chuẩn bảo mật...).",
        "question": "Có yêu cầu cụ thể nào về hiệu suất, bảo mật hay các ràng buộc kỹ thuật khác?",
    },
    {
        "id": "success_metrics",
        "name": "Chỉ số thành công",
        "description": "Tiêu chí định lượng hoặc định tính để đo lường mức độ thành công của dự án.",
        "score_0": "Không nêu ra cách nào để đánh giá thành công của dự án.",
        "score_5": "Xác định rõ các chỉ số hoặc tiêu chí đánh giá (ví dụ: số lượng người dùng, tỷ lệ lỗi, kinh doanh) cho thấy dự án thành công.",
        "question": "Dự án sẽ được đánh giá là thành công theo tiêu chí nào (KPIs)?",
    },
    {
        "id": "feasibility",
        "name": "Tính khả thi",
        "description": "Đánh giá về tính khả thi của dự án (về kỹ thuật, nguồn lực, thời gian, ngân sách).",
        "score_0": "Không có phân tích hoặc thông tin gì về tính khả thi của dự án.",
        "score_5": "Cung cấp phân tích chi tiết về khả năng thực hiện, bao gồm nguồn lực hiện có, thời gian, hạn chế về công nghệ.",
        "question": "Dự án này có những thách thức nào về mặt kỹ thuật hoặc tài nguyên?",
    },
    {
        "id": "risks_dependencies",
        "name": "Rủi ro và phụ thuộc",
        "description": "Các rủi ro tiềm ẩn và các yếu tố phụ thuộc bên ngoài (công nghệ, đối tác, quy định...).",
        "score_0": "Không xem xét bất kỳ rủi ro hoặc phụ thuộc nào.",
        "score_5": "Liệt kê rõ các rủi ro và phụ thuộc chính (với khả năng ảnh hưởng và biện pháp giảm thiểu).",
        "question": "Những rủi ro hoặc phụ thuộc nào có thể ảnh hưởng đến tiến độ hoặc chất lượng dự án?",
    },
]

# Tạo system prompt
SYSTEM_PROMPT = (
    """
Bạn là một AI Agent có nhiệm vụ đánh giá mức độ đầy đủ của yêu cầu dự án phần mềm.
Đánh giá theo 7 tiêu chí sau (điểm 0-5):

"""
    + "\n\n".join(
        [
            f"{idx+1}. {c['id']} ({c['name']}): {c['description']}\n- Điểm 0: {c['score_0']}\n- Điểm 5: {c['score_5']}\n- Nếu điểm thấp: Câu hỏi -> {c['question']}"
            for idx, c in enumerate(EVALUATION_CRITERIA)
        ]
    )
    + """

Yêu cầu:
1. Chấm điểm từng tiêu chí.
2. Nếu điểm tiêu chí ≤ 1, thêm câu hỏi tương ứng.
3. Tính điểm trung bình toàn bộ.

Trả về JSON theo đúng schema:
{
  "scores": {<id>: <score>, ...},
  "questions": [{"id": <id>, "question": <text>}],
  "overall_score": <float>
}
"""
)


# Agent class
class RequirementEvaluator:
    """Requirement evaluator agent"""

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        max_retries: int = 3,
    ):
        """Initialize the RequirementEvaluatorAgent.

        Args:
            model: The AI model to use for evaluation (default: "openai:gpt-4o")
            max_retries: Maximum number of retry attempts for API calls
        """
        self.model = model
        self.max_retries = max_retries
        self.agent = Agent(
            model,
            system_prompt=SYSTEM_PROMPT,
            output_type=EvaluationResult,
        )
        logger.info(f"Initialized RequirementEvaluatorAgent with model: {model}")

    async def evaluate(self, input_text: str) -> EvaluationResult:
        """Evaluate the input text against the evaluation criteria.

        Args:
            input_text: The project requirements text to evaluate

        Returns:
            EvaluationResult: The evaluation results with scores and questions

        Raises:
            ValueError: If input_text is empty or invalid
            RuntimeError: If evaluation fails after max_retries attempts
        """
        if not input_text or not isinstance(input_text, str):
            raise ValueError("Input text must be a non-empty string")

        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Evaluation attempt {attempt + 1}/{self.max_retries}")
                result = await self.agent.run(input_text)
                logger.info("Successfully evaluated requirements")
                return result.output

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    break

        logger.error(f"Evaluation failed after {self.max_retries} attempts")
        raise RuntimeError(
            f"Failed to evaluate requirements after {self.max_retries} attempts: {str(last_error)}"
        )


# def main():
#     """Example usage of the RequirementEvaluatorAgent."""

#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     )

#     try:
#         # Initialize the evaluator
#         agent = RequirementEvaluatorAgent()

#         # Sample requirements to evaluate
#         sample = (
#             "We want a plugin that, for every cake order, identifies the selected flavour "
#             "and cake weight, then automatically scales the recipe ingredient quantities, "
#             "frosting structure, and overall cost..."
#         )

#         # Evaluate the sample
#         logger.info("Starting evaluation...")
#         evaluation = agent.evaluate(sample)

#         # Print results
#         print("\n" + "=" * 50)
#         print("Requirements Evaluation Results")
#         print("=" * 50)
#         print(evaluation.model_dump_json(indent=2))

#     except Exception as e:
#         logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
#         return 1

#     return 0


# if __name__ == "__main__":
#     sys.exit(main())
