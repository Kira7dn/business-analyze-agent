"""
Question generator tool
"""

import logging
from typing import List
from ..models.schemas import QuestionGeneration

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generates clarification questions for requirements"""

    def __init__(self):
        self.question_templates = {
            "technical": [
                "What technology stack do you prefer for {}?",
                "What are the performance requirements for {}?",
                "Do you have any specific technical constraints for {}?",
                "What integrations are needed for {}?",
                "What is the expected scale/load for {}?",
            ],
            "business": [
                "Who is the target audience for {}?",
                "What is the primary business goal of {}?",
                "What is the expected timeline for {}?",
                "What is the budget range for {}?",
                "How will success be measured for {}?",
            ],
            "functional": [
                "What are the core features needed for {}?",
                "What user roles should be supported in {}?",
                "What workflows need to be implemented in {}?",
                "What data needs to be stored for {}?",
                "What reports/analytics are needed for {}?",
            ],
            "non-functional": [
                "What are the security requirements for {}?",
                "What are the availability requirements for {}?",
                "What are the scalability requirements for {}?",
                "What are the usability requirements for {}?",
                "What are the compliance requirements for {}?",
            ],
        }

    def generate(
        self, requirements: str, focus_area: str = "general"
    ) -> QuestionGeneration:
        """
        Generate clarification questions based on requirements

        Args:
            requirements: Project requirements description
            focus_area: Area to focus questions on (technical, business, functional, non-functional, general)

        Returns:
            QuestionGeneration object with generated questions
        """
        logger.info(f"Generating questions for focus area: {focus_area}")

        questions = self._generate_questions(requirements, focus_area)

        return QuestionGeneration(
            questions=questions, focus_area=focus_area, context=requirements
        )

    def _generate_questions(self, requirements: str, focus_area: str) -> List[str]:
        """Generate specific questions based on focus area"""
        questions = []

        # Extract key topics from requirements
        topics = self._extract_topics(requirements)

        if focus_area == "general":
            # Generate questions from all categories
            for category in self.question_templates:
                questions.extend(
                    self._apply_templates(category, topics[:2])
                )  # Limit topics
        else:
            # Generate questions for specific focus area
            if focus_area in self.question_templates:
                questions = self._apply_templates(focus_area, topics)
            else:
                # Fallback to general questions
                questions = self._generate_general_questions(requirements)

        return questions[:10]  # Limit to 10 questions

    def _extract_topics(self, requirements: str) -> List[str]:
        """Extract key topics from requirements text"""
        # Simple topic extraction - can be enhanced with NLP
        topics = []

        # Common project topics
        common_topics = [
            "user management",
            "authentication",
            "database",
            "api",
            "frontend",
            "backend",
            "mobile app",
            "web application",
            "dashboard",
            "reporting",
            "integration",
            "payment",
            "notification",
            "search",
            "analytics",
        ]

        requirements_lower = requirements.lower()
        for topic in common_topics:
            if topic in requirements_lower:
                topics.append(topic)

        # If no specific topics found, use generic terms
        if not topics:
            topics = ["the system", "the application", "the project"]

        return topics[:5]  # Limit to 5 topics

    def _apply_templates(self, category: str, topics: List[str]) -> List[str]:
        """Apply question templates to topics"""
        questions = []
        templates = self.question_templates.get(category, [])

        for template in templates:
            for topic in topics:
                question = template.format(topic)
                questions.append(question)

        return questions

    def _generate_general_questions(self, requirements: str) -> List[str]:
        """Generate general clarification questions"""
        return [
            "Can you provide more details about the core functionality?",
            "Who are the primary users of this system?",
            "What is the expected timeline for this project?",
            "Are there any specific technology preferences?",
            "What is the expected user load/scale?",
            "Are there any integration requirements?",
            "What are the security and compliance needs?",
            "What is the deployment environment?",
            "Are there any budget constraints?",
            "How will the success of this project be measured?",
        ]
