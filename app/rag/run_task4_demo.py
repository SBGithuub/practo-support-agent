"""
run_task4_demo.py - Runs Task 4 grounded generation demonstration on >=5 in-scope queries
plus 1 deliberately out-of-scope query triggering the fallback message.
"""

from app.rag.grounded_gen import generate_grounded_response, COLLECTION_SENTENCE
from app.config import SIMILARITY_THRESHOLD

TEST_QUERIES = [
    # 5 In-scope queries
    ("Q1", "What is the penalty for cancelling an appointment late inside the two-hour window?"),
    ("Q2", "How long do standard diagnostic lab tests like blood counts take to process?"),
    ("Q3", "What discount do I receive if I book a follow-up appointment within 7 days?"),
    ("Q4", "How can patients file for reimbursement of outpatient consultation fees?"),
    ("Q5", "Are home doctor visits available for bedridden elderly patients?"),
    # 1 Deliberately out-of-scope query
    ("Q6_OOS", "What is the proper baking temperature and recipe for chocolate fudge brownies?")
]

def run_task4_demo():
    print("=" * 70)
    print("TASK 4: GROUNDED GENERATION & CALIBRATED FALLBACK DEMONSTRATION")
    print(f"Empirical Similarity Fallback Threshold: {SIMILARITY_THRESHOLD}")
    print("=" * 70)
    
    for qid, query in TEST_QUERIES:
        print(f"\n--- [{qid}] ---")
        print(f"Query: '{query}'")
        res = generate_grounded_response(query, collection_name=COLLECTION_SENTENCE, use_cache=False)
        
        print(f"Top Retrieval Similarity : {res['top_similarity']:.4f}")
        print(f"Threshold Passed         : {res['top_similarity'] >= SIMILARITY_THRESHOLD}")
        print(f"Fallback Triggered       : {res['fallback_triggered']}")
        print(f"Sources Cited            : {res['sources']}")
        print(f"Answer Output:\n  \"{res['answer']}\"")
        
    print("\n" + "=" * 70)
    print("TASK 4 DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    run_task4_demo()
