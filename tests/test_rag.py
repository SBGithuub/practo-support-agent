import pytest
from app.config import KB_DIR, COLLECTION_SENTENCE, COLLECTION_FIXED, SIMILARITY_THRESHOLD
from app.rag.chunking import load_and_chunk_all, get_sentence_chunks, get_fixed_chunks
from app.rag.vector_store import query_collection
from app.rag.grounded_gen import generate_grounded_response, clear_cache, get_cache_stats
from app.rag.evaluate_rag import evaluate_collection

def test_chunking_strategies():
    fixed, sents = load_and_chunk_all(KB_DIR)
    assert len(fixed) > 0
    assert len(sents) > 0
    
    # Metadata preservation check
    for c in fixed:
        assert "doc_id" in c
        assert c["strategy"] == "fixed"
        assert len(c["text"]) > 0
        
    for c in sents:
        assert "doc_id" in c
        assert c["strategy"] == "sentence"
        assert len(c["text"]) > 0

def test_chroma_retrieval_in_scope():
    query = "What is the penalty for late appointment cancellation?"
    res = query_collection(COLLECTION_SENTENCE, query, top_k=2)
    assert len(res) > 0
    assert res[0]["doc_id"] == "cancellation_rescheduling.txt"
    assert res[0]["similarity"] > SIMILARITY_THRESHOLD

def test_chroma_retrieval_out_of_scope():
    query = "How do I replace the engine oil and spark plugs in a Honda Civic?"
    res = query_collection(COLLECTION_SENTENCE, query, top_k=1)
    assert len(res) > 0
    assert res[0]["similarity"] < SIMILARITY_THRESHOLD

def test_grounded_generation_in_scope():
    query = "How long do standard diagnostic lab test results take?"
    res = generate_grounded_response(query, use_cache=False)
    assert res["grounded"] is True
    assert res["fallback_triggered"] is False
    assert "lab_test_turnaround.txt" in res["sources"]
    assert "hours" in res["answer"].lower()

def test_grounded_generation_fallback():
    query = "What is the secret recipe for chocolate chip ice cream?"
    res = generate_grounded_response(query, use_cache=False)
    assert res["fallback_triggered"] is True
    assert res["grounded"] is False
    assert "do not have enough verified information" in res["answer"]

def test_response_caching():
    clear_cache()
    query = "What discount applies for follow up appointments within 7 days?"
    
    # First call - cache miss
    res1 = generate_grounded_response(query, use_cache=True)
    assert res1["cache_hit"] is False
    
    # Second call (with minor whitespace/case variation) - cache hit
    res2 = generate_grounded_response("  WHAT DISCOUNT APPLIES FOR FOLLOW UP APPOINTMENTS WITHIN 7 DAYS? ", use_cache=True)
    assert res2["cache_hit"] is True
    assert res2["answer"] == res1["answer"]
    
    stats = get_cache_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1

def test_rag_evaluation_metrics():
    metrics = evaluate_collection(COLLECTION_SENTENCE, top_k=2)
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["recall"] == 1.0

