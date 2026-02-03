"""
Contextual.ai Reranker integration for HyperGraphRAG.
"""
import os
import httpx
from typing import List, Dict, Any
from .utils import logger

def _load_api_key() -> str:
    """Load Contextual API key from environment or ~/.env file."""
    key = os.getenv("CONTEXTUAL_API_KEY")
    if not key:
        env_path = os.path.expanduser("~/.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("CONTEXTUAL_API_KEY="):
                        key = line.strip().split("=", 1)[1].strip('"\'')
                        break
    return key or ""

CONTEXTUAL_API_KEY = _load_api_key()
CONTEXTUAL_MODEL = "ctxl-rerank-v2-instruct-multilingual-mini"
CONTEXTUAL_URL = "https://api.contextual.ai/v1/rerank"


async def contextual_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    content_key: str = "entity_name",
    instruction: str = "",
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Rerank candidates using Contextual.ai API.

    Args:
        query: Search query
        candidates: List of dicts with content to rerank
        content_key: Key to extract text from each candidate
        instruction: Optional instruction to guide ranking
        top_k: Number of results to return

    Returns:
        Reranked list of candidates with added 'rerank_score'
    """
    if not CONTEXTUAL_API_KEY:
        logger.warning("CONTEXTUAL_API_KEY not found, skipping reranking")
        return candidates[:top_k]

    if not candidates:
        return []

    # Extract text from candidates
    documents = []
    for c in candidates:
        text = c.get(content_key, "") or c.get("content", "") or c.get("description", "") or str(c)
        documents.append(str(text)[:8000])  # Limit to 8K chars per doc

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                CONTEXTUAL_URL,
                headers={
                    "Authorization": f"Bearer {CONTEXTUAL_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "query": query,
                    "documents": documents,
                    "model": CONTEXTUAL_MODEL,
                    "instruction": instruction or "Retrieve relevant passages for the query"
                }
            )

        if response.status_code != 200:
            logger.warning(f"Contextual.ai error {response.status_code}: {response.text[:200]}")
            return candidates[:top_k]

        results = response.json().get("results", [])

        # Sort by score and rebuild list
        results_sorted = sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)

        reranked = []
        for item in results_sorted[:top_k]:
            idx = item["index"]
            if idx < len(candidates):
                candidate = candidates[idx].copy()
                candidate["rerank_score"] = item.get("relevance_score", 0)
                reranked.append(candidate)

        logger.info(f"Contextual.ai reranked {len(candidates)} -> {len(reranked)} candidates")
        return reranked

    except Exception as e:
        logger.warning(f"Contextual.ai exception: {e}")
        return candidates[:top_k]
