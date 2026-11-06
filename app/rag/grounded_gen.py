"""
grounded_gen.py - Grounded generation with empirical threshold fallback and response caching.
Supports Tasks 4, 10, and 16.
"""

import time
import re
from typing import Dict, Any, List, Tuple
from app.config import SIMILARITY_THRESHOLD, COLLECTION_SENTENCE
from app.rag.vector_store import query_collection
from app.guardrails.groundedness_check import check_groundedness, FALLBACK_MESSAGE

# In-memory query response cache (Task 16)
_RESPONSE_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_STATS = {"hits": 0, "misses": 0, "total_calls": 0}

def normalize_query(query: str) -> str:
    """Normalizes query text for deterministic cache lookup."""
    clean = re.sub(r'[^\w\s]', '', query.lower())
    return ' '.join(clean.split())

def clear_cache():
    """Clears the in-memory response cache."""
    global _RESPONSE_CACHE, _CACHE_STATS
    _RESPONSE_CACHE.clear()
    _CACHE_STATS = {"hits": 0, "misses": 0, "total_calls": 0}

def get_cache_stats() -> Dict[str, Any]:
    return dict(_CACHE_STATS)

def synthesize_grounded_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Deterministic grounded generation under MOCK_LLM.
    Strictly uses ONLY retrieved context sentences without introducing external facts.
    """
    if not retrieved_chunks:
        return FALLBACK_MESSAGE
        
    doc_ids = sorted(list(set(c["doc_id"] for c in retrieved_chunks)))
    top_text = retrieved_chunks[0]["text"]
    
    # Compose clean, grounded response directly citing the policy source
    answer = f"According to Practo's official policy documentation ({', '.join(doc_ids)}): {top_text}"
    if len(retrieved_chunks) > 1 and retrieved_chunks[1]["doc_id"] == retrieved_chunks[0]["doc_id"]:
        # Append corroborating context from same document
        if retrieved_chunks[1]["text"] not in answer:
            answer += f" Additionally, {retrieved_chunks[1]['text'].lower()}"
            
    return answer

def generate_grounded_response(
    query: str,
    collection_name: str = COLLECTION_SENTENCE,
    top_k: int = 3,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Performs vector retrieval, checks empirical threshold, applies caching, and generates response.
    """
    global _CACHE_STATS
    _CACHE_STATS["total_calls"] += 1
    norm_key = normalize_query(query)
    
    # Task 16: Check in-memory cache
    if use_cache and norm_key in _RESPONSE_CACHE:
        _CACHE_STATS["hits"] += 1
        cached_entry = _RESPONSE_CACHE[norm_key].copy()
        cached_entry["cache_hit"] = True
        return cached_entry
        
    _CACHE_STATS["misses"] += 1
    start_time = time.perf_counter()
    
    # 1. Retrieve top-k chunks
    retrieved = query_collection(collection_name, query, top_k=top_k)
    top_sim = retrieved[0]["similarity"] if retrieved else 0.0
    
    # 2. Check empirical similarity threshold (Task 4)
    if not retrieved or top_sim < SIMILARITY_THRESHOLD:
        response_data = {
            "query": query,
            "answer": FALLBACK_MESSAGE,
            "grounded": False,
            "fallback_triggered": True,
            "top_similarity": top_sim,
            "threshold": SIMILARITY_THRESHOLD,
            "sources": [],
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "cache_hit": False
        }
    else:
        # 3. Grounded generation from context
        draft_answer = synthesize_grounded_answer(query, retrieved)
        is_grounded, final_answer = check_groundedness(draft_answer, retrieved[0]["text"], top_sim, SIMILARITY_THRESHOLD)
        
        sources = list(dict.fromkeys(c["doc_id"] for c in retrieved))
        response_data = {
            "query": query,
            "answer": final_answer,
            "grounded": is_grounded,
            "fallback_triggered": not is_grounded,
            "top_similarity": top_sim,
            "threshold": SIMILARITY_THRESHOLD,
            "sources": sources,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "cache_hit": False
        }
        
    if use_cache:
        _RESPONSE_CACHE[norm_key] = response_data.copy()
        
    return response_data

