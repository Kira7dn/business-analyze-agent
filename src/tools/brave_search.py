"""
Brave Search API integration: News search agent and result model.
"""

import os
import logging
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import requests

# Setup logger
logger = logging.getLogger(__name__)

# System prompt for the LLM agent
SYSTEM_PROMPT = (
    "Bạn là một AI tìm kiếm thông tin trên web bằng Brave Search. Dựa vào truy vấn đầu vào, hãy trả về danh sách kết quả tìm kiếm "
    "bao gồm tiêu đề, URL, mô tả ngắn gọn và độ liên quan nếu có."
)


class SearchResult(BaseModel):
    """A news search result item, matching Brave NewsResult API."""

    type: Optional[str] = Field(
        None, description="Type of result (always 'news_result' for news)"
    )
    title: str = Field(..., description="Title of the news article")
    url: str = Field(..., description="Source URL of the news article")
    description: Optional[str] = Field(
        None, description="Description for the news article"
    )
    age: Optional[str] = Field(
        None, description="Human readable representation of the page age"
    )
    page_age: Optional[str] = Field(
        None, description="Page age from the source web page"
    )
    page_fetched: Optional[str] = Field(
        None, description="ISO date time when the page was last fetched"
    )
    breaking: Optional[bool] = Field(
        None, description="Whether the result includes breaking news"
    )
    thumbnail: Optional[str] = Field(None, description="URL of the thumbnail image")
    favicon: Optional[str] = Field(None, description="Favicon URL from meta_url")
    meta_url_path: Optional[str] = Field(
        None, description="Path from meta_url (for display)"
    )
    extra_snippets: Optional[list[str]] = Field(
        None, description="Extra alternate snippets"
    )


class BraveSearchAgent(Agent):
    """Agent for performing Brave web searches."""

    system_prompt: str = SYSTEM_PROMPT
    result_model = SearchResult

    def search(
        self, query: str, safesearch: str = "moderate", count: int = 10
    ) -> list[SearchResult]:
        """Perform a Brave Search API query and return a list of SearchResult."""
        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key:
            logger.error("Brave API key not set in environment variable BRAVE_API_KEY.")
            raise ValueError("Brave API key not set.")
        url = "https://api.search.brave.com/res/v1/news/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "x-subscription-token": api_key,
        }
        params = {"q": query, "safesearch": safesearch, "count": count}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("results", [])
            results = []
            for item in items:
                # Map Brave NewsResult API fields to SearchResult fields
                thumbnail = None
                if isinstance(item.get("thumbnail"), dict):
                    thumbnail = item["thumbnail"].get("src")
                elif isinstance(item.get("thumbnail"), str):
                    thumbnail = item.get("thumbnail")
                meta_url = item.get("meta_url", {})
                # Only process items of type 'news_result' for safety
                if item.get("type") != "news_result":
                    continue
                results.append(
                    SearchResult(
                        type=item.get("type"),
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        description=item.get("description"),
                        age=item.get("age"),
                        page_age=item.get("page_age"),
                        page_fetched=item.get("page_fetched"),
                        breaking=item.get("breaking"),
                        thumbnail=thumbnail,
                        favicon=meta_url.get("favicon"),
                        meta_url_path=meta_url.get("path"),
                        extra_snippets=item.get("extra_snippets"),
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Brave Search API error: {e}")
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    QUERY = "warehouse software"
    agent = BraveSearchAgent()
    print(f"Searching Brave News for: {QUERY}")
    results = agent.search(QUERY)
    for idx, result in enumerate(results, 1):
        print(f"\nResult {idx}:")
        print(result.model_dump_json(indent=2))
