import os
from typing import List, Dict, Any
from blog_generator.interfaces import SearchService

class TavilySearchService(SearchService):
    """
    Tavily search service implementation using asynchronous execution.
    """
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        # Respect the same fallback check in the original logic
        if not os.getenv("TAVILY_API_KEY"):
            return []
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
            tool = TavilySearchResults(max_results=max_results)
            # Invoke the tool asynchronously
            results = await tool.ainvoke({"query": query})
            out: List[Dict[str, Any]] = []
            for r in results or []:
                out.append(
                    {
                        "title": r.get("title") or "",
                        "url": r.get("url") or "",
                        "snippet": r.get("content") or r.get("snippet") or "",
                        "published_at": r.get("published_date") or r.get("published_at"),
                        "source": r.get("source"),
                    }
                )
            return out
        except Exception:
            return []
