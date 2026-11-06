"""
groundedness_check.py - Output-side groundedness verification.
Refuses to answer when the retrieved context does not support the response.
"""

FALLBACK_MESSAGE = (
    "I apologize, but I do not have enough verified information in Practo's official "
    "policy documentation to answer your question accurately. Please contact Practo clinic support "
    "at support@practo.com or call our patient helpline directly."
)

def check_groundedness(draft_answer: str, retrieved_context: str, similarity_score: float, threshold: float = 0.45) -> tuple[bool, str]:
    """
    Validates output groundedness against retrieved context and similarity score.
    Returns (is_grounded, final_text).
    """
    if not retrieved_context or not retrieved_context.strip():
        return False, FALLBACK_MESSAGE
        
    if similarity_score < threshold:
        return False, FALLBACK_MESSAGE
        
    # Check if the draft is an explicit admission of lack of info
    if "do not have enough" in draft_answer.lower() or "not covered in" in draft_answer.lower():
        return False, FALLBACK_MESSAGE
        
    return True, draft_answer
