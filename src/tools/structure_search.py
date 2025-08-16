"""
RAG Agent module: tech structure suggestion using Pydantic AI.
"""

from pydantic_ai import Agent

from src.config.settings import settings

from .utils import ContextRetriever


class StructureSearch:
    """Agent for suggesting tech structure for a project, depends on context retriever and AI agent."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a senior software architect. Based on the following project requirements, "
        "propose a complete yet minimal tech structure optimized for performance and memory constraints:\n\n"
        "{context}"
    )

    def __init__(
        self,
        ai_agent: Agent = None,
        context_retriever: ContextRetriever = None,
    ):
        self.ai_agent = ai_agent or Agent(
            model=settings.model_choice,
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )
        self.context_retriever = context_retriever or ContextRetriever()

    def suggest(self, query: str, match_count: int = 3):
        """
        Suggest a tech structure for a given project requirement query.
        Args:
            query (str): Project requirement query.
            match_count (int): Number of context documents to retrieve.
        Returns:
            Kết quả sinh ra từ agent (text hoặc dict)
        """
        context = self.context_retriever.get_context(query, match_count)
        if not context or not context.strip():
            raise ValueError("No context available for the given query.")
        prompt = f"{query}\n\nContext:\n{context}"
        result = self.ai_agent.run_sync(prompt)
        return result.output
