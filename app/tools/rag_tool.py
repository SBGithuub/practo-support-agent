"""
rag_tool.py - Knowledge base retrieval tool for CrewAI Retrieval Agent.
"""

from crewai.tools import tool
from app.rag.grounded_gen import generate_grounded_response
from app.config import COLLECTION_SENTENCE

@tool("rag_policy_lookup")
def rag_policy_lookup(query: str) -> str:
    """
    Search Practo clinic policies, consultation fees, discounts, and guidelines.
    Input argument: query (str) - the policy question to search.
    """
    result = generate_grounded_response(query, collection_name=COLLECTION_SENTENCE)
    if result["fallback_triggered"]:
        return result["answer"]
        
    sources = ", ".join(result["sources"]) if result["sources"] else "General Policy"
    return f"Retrieved Context from {sources}: {result['answer']}"
