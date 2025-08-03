"""
Requirements analyzer tool
"""

import logging
from typing import List, Dict
from ..models.schemas import RequirementAnalysis

logger = logging.getLogger(__name__)


class RequirementAnalyzer:
    """Analyzes project requirements using MoSCoW method"""

    def __init__(self):
        self.project_types = {
            "web": ["frontend", "backend", "database", "authentication", "deployment"],
            "mobile": [
                "ui/ux",
                "platform",
                "offline capability",
                "push notifications",
                "app store",
            ],
            "api": [
                "endpoints",
                "authentication",
                "rate limiting",
                "documentation",
                "versioning",
            ],
            "desktop": [
                "ui framework",
                "cross-platform",
                "installation",
                "updates",
                "performance",
            ],
            "other": [
                "architecture",
                "scalability",
                "security",
                "maintenance",
                "integration",
            ],
        }

    def analyze(
        self, description: str, project_type: str = "web"
    ) -> RequirementAnalysis:
        """
        Analyze project requirements and categorize using MoSCoW method

        Args:
            description: Project description
            project_type: Type of project (web, mobile, api, desktop, other)

        Returns:
            RequirementAnalysis object with categorized requirements
        """
        logger.info(f"Analyzing requirements for {project_type} project")

        # Extract requirements from description
        requirements = self._extract_requirements(description, project_type)

        # Categorize using MoSCoW method
        moscow_priority = self._categorize_moscow(requirements, project_type)

        # Identify gaps and suggestions
        gaps = self._identify_gaps(requirements, project_type)
        suggestions = self._generate_suggestions(requirements, project_type)

        return RequirementAnalysis(
            requirements=requirements,
            moscow_priority=moscow_priority,
            gaps=gaps,
            suggestions=suggestions,
        )

    def _extract_requirements(self, description: str, project_type: str) -> List[str]:
        """Extract requirements from project description"""
        requirements = []

        # Basic keyword extraction
        keywords = description.lower().split()

        # Add project type specific requirements
        if project_type in self.project_types:
            for area in self.project_types[project_type]:
                if any(keyword in description.lower() for keyword in area.split()):
                    requirements.append(f"{area.title()} implementation")

        # Extract explicit requirements
        sentences = description.split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                requirements.append(sentence)

        return list(set(requirements))  # Remove duplicates

    def _categorize_moscow(
        self, requirements: List[str], project_type: str
    ) -> Dict[str, List[str]]:
        """Categorize requirements using MoSCoW method"""
        moscow = {"must": [], "should": [], "could": [], "wont": []}

        # Basic categorization logic
        critical_keywords = [
            "authentication",
            "security",
            "database",
            "core",
            "essential",
            "required",
        ]
        important_keywords = [
            "ui",
            "user interface",
            "performance",
            "optimization",
            "integration",
        ]
        nice_keywords = ["analytics", "reporting", "advanced", "additional", "extra"]

        for req in requirements:
            req_lower = req.lower()

            if any(keyword in req_lower for keyword in critical_keywords):
                moscow["must"].append(req)
            elif any(keyword in req_lower for keyword in important_keywords):
                moscow["should"].append(req)
            elif any(keyword in req_lower for keyword in nice_keywords):
                moscow["could"].append(req)
            else:
                moscow["should"].append(req)  # Default to should

        return moscow

    def _identify_gaps(self, requirements: List[str], project_type: str) -> List[str]:
        """Identify potential gaps in requirements"""
        gaps = []

        # Check for common missing requirements by project type
        if project_type == "web":
            if not any("security" in req.lower() for req in requirements):
                gaps.append("Security considerations not explicitly mentioned")
            if not any("responsive" in req.lower() for req in requirements):
                gaps.append("Responsive design requirements unclear")

        elif project_type == "mobile":
            if not any("platform" in req.lower() for req in requirements):
                gaps.append("Target platform (iOS/Android) not specified")
            if not any("offline" in req.lower() for req in requirements):
                gaps.append("Offline functionality requirements missing")

        return gaps

    def _generate_suggestions(
        self, requirements: List[str], project_type: str
    ) -> List[str]:
        """Generate suggestions for improvement"""
        suggestions = []

        # General suggestions
        suggestions.append("Consider implementing comprehensive error handling")
        suggestions.append("Plan for scalability and future growth")
        suggestions.append("Include monitoring and logging capabilities")

        # Project type specific suggestions
        if project_type == "web":
            suggestions.append("Implement SEO optimization")
            suggestions.append("Consider Progressive Web App (PWA) features")
        elif project_type == "mobile":
            suggestions.append("Plan for app store optimization")
            suggestions.append("Consider push notification strategy")

        return suggestions
