#!/usr/bin/env python3
"""
Business Analyze Agent - MCP Server
Basic MCP server for business requirements analysis
"""

import asyncio
import logging
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("business-analyze-agent")


class RequirementAnalysis(BaseModel):
    """Model for requirement analysis results"""

    requirements: List[str]
    moscow_priority: Dict[str, List[str]]
    gaps: List[str]
    suggestions: List[str]


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="analyze_requirements",
            description="Analyze project requirements and categorize them using MoSCoW method",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_description": {
                        "type": "string",
                        "description": "Description of the project requirements",
                    },
                    "project_type": {
                        "type": "string",
                        "description": "Type of project (web, mobile, api, etc.)",
                        "enum": ["web", "mobile", "api", "desktop", "other"],
                    },
                },
                "required": ["project_description"],
            },
        ),
        Tool(
            name="generate_questions",
            description="Generate clarification questions based on project requirements",
            inputSchema={
                "type": "object",
                "properties": {
                    "requirements": {
                        "type": "string",
                        "description": "Project requirements to analyze",
                    },
                    "focus_area": {
                        "type": "string",
                        "description": "Specific area to focus questions on",
                        "enum": ["technical", "functional", "business", "all"],
                    },
                },
                "required": ["requirements"],
            },
        ),
        Tool(
            name="health_check",
            description="Check if the MCP server is running properly",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""

    if name == "health_check":
        return [
            TextContent(
                type="text",
                text="✅ Business Analyze Agent MCP Server is running!\n\nAvailable tools:\n- analyze_requirements\n- generate_questions\n- health_check",
            )
        ]

    elif name == "analyze_requirements":
        project_description = arguments.get("project_description", "")
        project_type = arguments.get("project_type", "web")

        # Basic requirement analysis logic
        analysis = analyze_project_requirements(project_description, project_type)

        result = f"""
# Requirements Analysis Report

## Project Type: {project_type.upper()}

## Identified Requirements:
{chr(10).join(f"- {req}" for req in analysis.requirements)}

## MoSCoW Prioritization:

### Must Have:
{chr(10).join(f"- {req}" for req in analysis.moscow_priority.get('must', []))}

### Should Have:
{chr(10).join(f"- {req}" for req in analysis.moscow_priority.get('should', []))}

### Could Have:
{chr(10).join(f"- {req}" for req in analysis.moscow_priority.get('could', []))}

### Won't Have (this time):
{chr(10).join(f"- {req}" for req in analysis.moscow_priority.get('wont', []))}

## Identified Gaps:
{chr(10).join(f"- {gap}" for gap in analysis.gaps)}

## Suggestions:
{chr(10).join(f"- {suggestion}" for suggestion in analysis.suggestions)}
"""

        return [TextContent(type="text", text=result)]

    elif name == "generate_questions":
        requirements = arguments.get("requirements", "")
        focus_area = arguments.get("focus_area", "all")

        questions = generate_clarification_questions(requirements, focus_area)

        result = f"""
# Clarification Questions ({focus_area.upper()})

{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(questions))}

---
*These questions will help clarify requirements and reduce project risks.*
"""

        return [TextContent(type="text", text=result)]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


def analyze_project_requirements(
    description: str, project_type: str
) -> RequirementAnalysis:
    """Basic requirement analysis logic"""

    # Simple keyword-based analysis (will be enhanced later)
    requirements = []
    moscow_priority = {"must": [], "should": [], "could": [], "wont": []}
    gaps = []
    suggestions = []

    # Extract basic requirements
    if "user" in description.lower():
        requirements.append("User management system")
        moscow_priority["must"].append("User authentication")
        moscow_priority["should"].append("User profile management")

    if "database" in description.lower() or "data" in description.lower():
        requirements.append("Database integration")
        moscow_priority["must"].append("Data persistence")
        moscow_priority["should"].append("Data backup strategy")

    if "api" in description.lower():
        requirements.append("API development")
        moscow_priority["must"].append("RESTful API endpoints")
        moscow_priority["could"].append("API documentation")

    # Project type specific requirements
    if project_type == "web":
        requirements.extend(["Frontend interface", "Backend server"])
        moscow_priority["must"].extend(
            ["Responsive design", "Cross-browser compatibility"]
        )
        suggestions.extend(["Consider NextJS for frontend", "FastAPI for backend"])

    # Identify common gaps
    if "security" not in description.lower():
        gaps.append("Security requirements not specified")

    if "performance" not in description.lower():
        gaps.append("Performance requirements unclear")

    if "deployment" not in description.lower():
        gaps.append("Deployment strategy not mentioned")

    # Default suggestions
    suggestions.extend(
        [
            "Define clear acceptance criteria",
            "Consider scalability requirements",
            "Plan for testing strategy",
        ]
    )

    return RequirementAnalysis(
        requirements=requirements,
        moscow_priority=moscow_priority,
        gaps=gaps,
        suggestions=suggestions,
    )


def generate_clarification_questions(requirements: str, focus_area: str) -> List[str]:
    """Generate clarification questions based on requirements and focus area"""

    questions = []
    req_lower = requirements.lower()

    if focus_area in ["technical", "all"]:
        questions.extend(
            [
                "What is the expected number of concurrent users?",
                "Are there any specific technology preferences or constraints?",
                "What are the performance requirements (response time, throughput)?",
                "Do you need real-time features or is eventual consistency acceptable?",
            ]
        )

        # Add context-specific technical questions
        if "database" in req_lower or "data" in req_lower:
            questions.append("What type of database do you prefer (SQL/NoSQL)?")
        if "api" in req_lower:
            questions.append("Do you need REST API, GraphQL, or both?")

    if focus_area in ["functional", "all"]:
        questions.extend(
            [
                "What are the main user roles and their permissions?",
                "What are the core workflows users need to complete?",
                "Are there any integration requirements with existing systems?",
                "What data needs to be stored and how long should it be retained?",
            ]
        )

        # Add context-specific functional questions
        if "user" in req_lower:
            questions.append("What user authentication method do you prefer?")
        if "payment" in req_lower or "commerce" in req_lower:
            questions.append("Which payment gateways need to be supported?")

    if focus_area in ["business", "all"]:
        questions.extend(
            [
                "What is the target launch date and are there any hard deadlines?",
                "What is the budget range for this project?",
                "Who are the main stakeholders and decision makers?",
                "What defines success for this project?",
            ]
        )

        # Add context-specific business questions
        if "scale" in req_lower or "growth" in req_lower:
            questions.append("What is your expected user growth over the next year?")

    return questions[:8]  # Limit to 8 questions


async def main():
    """Main entry point"""
    logger.info("Starting Business Analyze Agent MCP Server...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
