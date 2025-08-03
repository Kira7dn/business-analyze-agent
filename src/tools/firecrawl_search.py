"""
Firecrawl API integration: Web scraping agent and result model.
"""

import os
import logging
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from firecrawl import FirecrawlApp

# Setup logger
logger = logging.getLogger(__name__)

# System prompt for the LLM agent
SYSTEM_PROMPT = (
    "Bạn là một AI thu thập và trích xuất dữ liệu web bằng Firecrawl API. Dựa vào URL đầu vào, hãy trả về thông tin đã trích xuất "
    "bao gồm tiêu đề, mô tả, nội dung chính và metadata nếu có."
)


class FirecrawlResult(BaseModel):
    """A web scrape result item, matching Firecrawl API output."""

    url: str = Field(..., description="Source URL of the scraped page")
    title: Optional[str] = Field(None, description="Title of the page")
    description: Optional[str] = Field(None, description="Page description")
    markdown: Optional[str] = Field(None, description="Markdown content if available")
    html: Optional[str] = Field(None, description="HTML content if available")
    metadata: Optional[dict] = Field(None, description="Metadata from Firecrawl scrape")
    actions: Optional[dict] = Field(
        None, description="Actions (screenshots, scrapes, etc.)"
    )


class FirecrawlAgent:
    """Agent for scraping web pages using Firecrawl API (firecrawl-py SDK), using composition instead of subclassing Agent."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.agent = Agent(system_prompt=system_prompt)
        self.result_model = FirecrawlResult

    def scrape(
        self, url: str, api_key: Optional[str] = None
    ) -> Optional[FirecrawlResult]:
        """Scrape a web page using Firecrawl Cloud API and return a FirecrawlResult."""
        api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            logger.error(
                "Firecrawl API key not set in environment variable FIRECRAWL_API_KEY."
            )
            raise ValueError("Firecrawl API key not set.")
        try:
            app = FirecrawlApp(api_key=api_key)
            scrape_result = app.scrape_url(url, formats=["markdown", "html"])
            d = getattr(scrape_result, "data", {})
            return FirecrawlResult(
                url=url,
                title=d.get("metadata", {}).get("title"),
                description=d.get("metadata", {}).get("description"),
                markdown=d.get("markdown"),
                html=d.get("html"),
                metadata=d.get("metadata"),
                actions=d.get("actions"),
            )
        except Exception as e:
            logger.error(f"Firecrawl SDK error: {e}")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    SAMPLE_URL = "https://docs.firecrawl.dev/features/crawl#installation"
    agent = FirecrawlAgent()
    print(f"Scraping with Firecrawl: {SAMPLE_URL}")
    result = agent.scrape(SAMPLE_URL)
    if result:
        print(result.model_dump_json(indent=2))
    else:
        print("No result or error occurred.")
