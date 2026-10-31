import asyncio
from datetime import date, timedelta
from typing import List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from blog_generator.config import llm
from blog_generator.prompts import RESEARCH_SYSTEM
from blog_generator.models import State, EvidencePack
from blog_generator.services.factory import ServiceFactory
from blog_generator.interfaces import SearchService

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

def research_node(state: State, search_service: Optional[SearchService] = None) -> dict:
    """
    Research node that queries search service concurrently for all queries.
    Utilizes a synchronous entry point that runs the async concurrency under asyncio.run.
    """
    if search_service is None:
        search_service = ServiceFactory.get_search_service()

    queries = (state.get("queries") or [])[:10]
    
    async def run_async_research() -> EvidencePack:
        # Trigger all search tasks concurrently
        search_tasks = [search_service.search(q, max_results=6) for q in queries]
        search_results = await asyncio.gather(*search_tasks)

        raw: List[dict] = []
        for results in search_results:
            raw.extend(results)

        if not raw:
            return EvidencePack(evidence=[])

        extractor = llm.with_structured_output(EvidencePack)
        pack = await extractor.ainvoke(
            [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(
                    content=(
                        f"As-of date: {state['as_of']}\n"
                        f"Recency days: {state['recency_days']}\n\n"
                        f"Raw results:\n{raw}"
                    )
                ),
            ]
        )
        return pack

    pack = asyncio.run(run_async_research())

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}
