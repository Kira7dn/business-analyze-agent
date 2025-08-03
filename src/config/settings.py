"""
Application configuration using Pydantic Settings
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration"""

    next_public_api_url: str = "http://localhost:8000"
    database_url: str = "postgresql://user:password@localhost:5432/dbname"
    node_env: str = "development"
    openai_api_key: str = ""
    brave_api_key: str = ""
    firecrawl_api_key: str = ""
    supabase_url: str = ""
    supabase_key: str = ""
    server_name: str = "business-analyze-agent"


# Global settings instance
settings = Settings()
