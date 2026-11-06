"""
calibrate_threshold.py - Measures top-1 cosine similarity for in-scope and out-of-scope queries
to empirically calibrate the grounded generation fallback threshold (Task 4).
"""

from app.rag.vector_store import query_collection, COLLECTION_FIXED, COLLECTION_SENTENCE

IN_SCOPE_QUERIES = [
    "What is the penalty for cancelling an appointment late?",
    "How fast are standard blood lab test results processed?",
    "Is there a discount on follow-up doctor visits within 7 days?",
    "How does cashless health insurance work at network hospitals?",
    "Are home healthcare visits available for bedridden elderly patients?"
]

OUT_OF_SCOPE_QUERIES = [
    "What ingredients are needed to bake chocolate chip cookies?",
    "How do I replace the engine oil and filter in a Toyota car?",
    "Who won the FIFA football world cup tournament in 2022?"
]

def run_calibration():
    print("=" * 65)
    print("EMPIRICAL CALIBRATION OF RETRIEVAL SIMILARITY THRESHOLD (TASK 4)")
    print("=" * 65)
    
    for coll_name in [COLLECTION_SENTENCE, COLLECTION_FIXED]:
        print(f"\nEvaluating Collection: '{coll_name}'")
        print("-" * 55)
        
        in_scope_scores = []
        print("In-Scope Queries:")
        for q in IN_SCOPE_QUERIES:
            results = query_collection(coll_name, q, top_k=1)
            top_sim = results[0]["similarity"] if results else 0.0
            doc_id = results[0]["doc_id"] if results else "none"
            in_scope_scores.append(top_sim)
            print(f"  [IN-SCOPE] '{q}'\n    -> Top Doc: {doc_id} | Similarity: {top_sim:.4f}")
            
        out_scope_scores = []
        print("\nOut-of-Scope Queries:")
        for q in OUT_OF_SCOPE_QUERIES:
            results = query_collection(coll_name, q, top_k=1)
            top_sim = results[0]["similarity"] if results else 0.0
            doc_id = results[0]["doc_id"] if results else "none"
            out_scope_scores.append(top_sim)
            print(f"  [OUT-OF-SCOPE] '{q}'\n    -> Top Doc: {doc_id} | Similarity: {top_sim:.4f}")
            
        min_in = min(in_scope_scores)
        max_in = max(in_scope_scores)
        avg_in = sum(in_scope_scores) / len(in_scope_scores)
        
        min_out = min(out_scope_scores)
        max_out = max(out_scope_scores)
        avg_out = sum(out_scope_scores) / len(out_scope_scores)
        
        print(f"\nSummary for '{coll_name}':")
        print(f"  In-Scope Range    : {min_in:.4f} to {max_in:.4f} (Avg: {avg_in:.4f})")
        print(f"  Out-of-Scope Range: {min_out:.4f} to {max_out:.4f} (Avg: {avg_out:.4f})")
        
        # Empirical midpoint threshold
        recommended_threshold = round((min_in + max_out) / 2.0, 4)
        print(f"  Recommended Threshold (midpoint between {min_in:.4f} and {max_out:.4f}): {recommended_threshold:.4f}")
        
    print("=" * 65)

if __name__ == "__main__":
    run_calibration()
