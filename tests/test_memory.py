"""
test_memory.py - Unit tests for LangChain session-based memory across turns and resets.
"""

import pytest
from app.agents.memory import ConversationalSessionManager, get_session_history, clear_session_history

def mock_turn_executor(query: str, history_context: str) -> str:
    if "follow-up" in query.lower() or "it" in query.lower():
        if "APT-0007" in history_context:
            return "Referencing appointment APT-0007 from earlier context: follow-up is required."
        return "Please specify which appointment you are referring to."
    return f"Processed query: {query}"

def test_session_memory_continuity():
    session_id = "test_turn_continuity"
    clear_session_history(session_id)
    
    manager = ConversationalSessionManager(mock_turn_executor)
    
    # Turn 1: mentions APT-0007
    r1 = manager.chat(session_id, "Checking APT-0007 status.")
    assert "Processed query: Checking APT-0007 status." in r1
    
    # Turn 2: anaphoric query
    r2 = manager.chat(session_id, "Is follow-up required for it?")
    assert "Referencing appointment APT-0007 from earlier context" in r2
    
    history = get_session_history(session_id)
    assert len(history.messages) == 4

def test_fresh_session_isolation():
    fresh_id = "test_fresh_isolation"
    clear_session_history(fresh_id)
    
    manager = ConversationalSessionManager(mock_turn_executor)
    
    # Fresh session with anaphoric query has no context
    r = manager.chat(fresh_id, "Is follow-up required for it?")
    assert "Please specify which appointment" in r
    
    history = get_session_history(fresh_id)
    assert len(history.messages) == 2
