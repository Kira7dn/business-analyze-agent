#!/usr/bin/env python3
"""
Business Analyze Agent - MCP Server
Restructured MCP server for business requirements analysis
"""

import asyncio
from typing import Any, Dict, List
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from dotenv import load_dotenv

from .tools.chunk_embed_store import ChunkEmbedStore
from .tools.stack_search import StackSearch
from .tools.analyzer import RequirementAnalyzer
from .tools.question_generator import QuestionGenerator
from .tools.requirement_evaluator import RequirementEvaluator
from .tools.feature_suggestion import FeatureSuggestionAgent
from .tools.be_object_parser import BEClassParser
from .tools.component_designer import ComponentDesigner
from .utils.logger import setup_logger

# Setup logging
load_dotenv()
logger = setup_logger(__name__)
# Initialize MCP server
server = Server("business-analyze-agent")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY: ", OPENAI_API_KEY)

# Initialize tools
try:
    analyzer = RequirementAnalyzer()
    question_gen = QuestionGenerator()
    evaluator = RequirementEvaluator()
    feature_suggester = FeatureSuggestionAgent()
    stack_search = StackSearch()
    stack_store = ChunkEmbedStore(table_name="tech_stacks")
    structure_store = ChunkEmbedStore(table_name="tech_structures")
    be_class_parser = BEClassParser()
    component_designer = ComponentDesigner()
    logger.info("All tools initialized successfully")

except Exception as exc:  # noqa: BLE001
    logger.exception("Failed to initialize RequirementEvaluator: %s", exc)


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="evaluate_requirements",
            description="Evaluate project requirements against quality criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "requirements": {
                        "type": "string",
                        "description": "Project requirements text to evaluate",
                    }
                },
                "required": ["requirements"],
            },
        ),
        Tool(
            name="health_check",
            description="Check if the MCP server is running and list available tools",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="suggest_features",
            description="Suggest features by user role and priority for a given project description.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_description": {
                        "type": "string",
                        "description": "Description of the project for which to suggest features.",
                    }
                },
                "required": ["project_description"],
            },
        ),
        Tool(
            name="stack_search",
            description="Suggest tech stack for a given project requirement query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Project requirement query.",
                    }
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="stack_store",
            description="Store chunks of text into Supabase with embeddings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to store.",
                    }
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="structure_store",
            description="Store chunks of text into Supabase with embeddings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to store.",
                    }
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="be_class_parser",
            description="Parse PRD text to backend classes using Pydantic AI.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prd_text": {
                        "type": "string",
                        "description": "Product Requirements Document text to parse.",
                    }
                },
                "required": ["prd_text"],
            },
        ),
        Tool(
            name="component_designer",
            description="Generate Clean Architecture components from PRD text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prd_text": {
                        "type": "string",
                        "description": "Product Requirements Document text to transform into components.",
                    }
                },
                "required": ["prd_text"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""

    try:
        if name == "health_check":
            return [
                TextContent(
                    type="text",
                    text="✅ Business Analyze Agent MCP Server is running!",
                )
            ]

        elif name == "analyze_requirements":
            project_description = arguments.get("project_description", "")
            project_type = arguments.get("project_type", "web")

            logger.info("Analyzing requirements for %s project", project_type)

            # Analyze requirements using the analyzer tool
            analysis = analyzer.analyze(project_description, project_type)

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
            focus_area = arguments.get("focus_area", "general")

            logger.info("Generating questions for focus area: %s", focus_area)

            # Generate questions using the question generator tool
            question_result = question_gen.generate(requirements, focus_area)

            result = f"""
                # Clarification Questions

                ## Focus Area: {focus_area.upper()}

                ## Generated Questions:
                {chr(10).join(f"{i+1}. {question}" for i, question in enumerate(question_result.questions))}

                ## Context:
                {question_result.context[:200]}{'...' if len(question_result.context) > 200 else ''}
            """

            return [TextContent(type="text", text=result)]

        elif name == "evaluate_requirements":
            requirements = arguments.get("requirements", "")

            logger.info("Evaluating requirements quality")

            # Evaluate requirements using the evaluator tool
            evaluation = await evaluator.evaluate(requirements)

            # Format the evaluation results
            result = "# Requirements Quality Evaluation\n\n"

            # Add scores summary
            result += "## Evaluation Scores\n"
            for criterion, score in evaluation.scores.items():
                result += f"- **{criterion.replace('_', ' ').title()}**: {score}/5\n"

            # Add overall score
            result += f"\n## Overall Score: {evaluation.overall_score:.1f}/5\n\n"

            # Add questions for improvement
            if evaluation.questions:
                result += "## Questions for Improvement\n"
                for question in evaluation.questions:
                    result += f"- {question.question}\n"

            return [TextContent(type="text", text=result)]

        elif name == "suggest_features":
            project_description = arguments.get("project_description", "")
            logger.info("Suggesting features for project description")
            suggestions = await feature_suggester.suggest(project_description)
            # Format output as markdown
            result = "# Feature Suggestions by User Role\n\n"
            for role_features in suggestions.suggestions:
                result += f"## Role: {role_features.role}\n"
                for feat in role_features.features:
                    result += (
                        f"- **{feat.name}**"
                        f" ({feat.priority})"
                        f"{f': {feat.description}' if feat.description else ''}\n"
                    )
                result += "\n"
            return [TextContent(type="text", text=result)]
        elif name == "stack_search":
            query = arguments.get("query", "")
            logger.info("Searching for tech stack")
            stack = await stack_search.suggest(query)
            return [TextContent(type="text", text=stack)]
        elif name == "stack_store":
            file_path = arguments.get("file_path", "")
            logger.info(
                "Storing chunks of text into Supabase with embeddings: %s", file_path
            )
            result = await stack_store.process(file_path)
            return [TextContent(type="text", text=result)]
        elif name == "structure_store":
            file_path = arguments.get("file_path", "")
            logger.info(
                "Storing chunks of text into Supabase with embeddings: %s", file_path
            )
            result = await structure_store.process(file_path)
            return [TextContent(type="text", text=result)]
        elif name == "be_class_parser":
            prd_text = arguments.get("prd_text", "")
            logger.info("Parsing PRD to backend classes")
            # be_class_parser = BEClassParser()
            result = await be_class_parser.process(prd_text)
            return [TextContent(type="text", text=result)]
        elif name == "component_designer":
            prd_text = arguments.get("prd_text", "")
            if not prd_text:
                logger.warning("component_designer called without prd_text")
                return [
                    TextContent(
                        type="text",
                        text="❌ Missing 'prd_text' argument for component_designer tool.",
                    )
                ]

            logger.info("Generating components from PRD text")
            result = await component_designer.process(prd_text)
            return [TextContent(type="text", text=result)]
        else:
            return [
                TextContent(
                    type="text",
                    text=f"Unknown tool: {name}",
                )
            ]

    except Exception as e:
        logger.error("Error in tool %s: %s", name, e)
        return [
            TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}",
            )
        ]


async def main():
    """Main entry point"""
    logger.info("Starting MCP Server")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
