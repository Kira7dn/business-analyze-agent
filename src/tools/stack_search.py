"""
RAG Agent module: tech stack suggestion using Pydantic AI.
"""

import os
import openai
from dotenv import load_dotenv
from supabase import create_client, Client
from pydantic_ai import Agent

# ===========================
# Config & Environment Loader
# ===========================


class Config:
    """Load and validate environment variables for the application."""

    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in environment.")


config = Config()
openai.api_key = config.openai_api_key

# ===========================
# Utility Functions
# ===========================


# ===========================
# Pydantic Model & AI Decorator
# ===========================


class ContextRetriever:
    """Retrieve context from Supabase database."""

    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)

    def get_context(self, query: str, match_count: int = 3) -> str:
        """
        Retrieve context from Supabase database based on semantic similarity to the query.
        Args:
            query (str): Query string for semantic search.
            match_count (int): Number of context documents to retrieve.
        Returns:
            str: Retrieved context.
        """
        try:
            # 1. Generate embedding for the query
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=query,
            )
            query_embedding = response.data[0].embedding

            # 2. Call the match_tech_stacks function via Supabase RPC
            response = self.supabase.rpc(
                "match_tech_stacks",
                {
                    "query_embedding": query_embedding,
                    "match_count": match_count,
                },
            ).execute()
            results = [item["content"] for item in response.data]
            return "\n\n".join(results)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve context: {e}") from e


class StackSearch:
    """Agent for suggesting tech stack for a project, depends on context retriever and AI agent."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a senior software architect. Based on the following project requirements, "
        "propose a complete yet minimal tech stack optimized for performance and memory constraints:\n\n"
        "{context}"
    )

    def __init__(
        self,
        ai_agent: Agent = None,
        context_retriever: ContextRetriever = None,
    ):
        self.ai_agent = ai_agent or Agent(
            model="openai:gpt-4o",
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
        )
        self.context_retriever = context_retriever or ContextRetriever(
            os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
        )

    def suggest(self, query: str, match_count: int = 3):
        """
        Suggest a tech stack for a given project requirement query.
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
