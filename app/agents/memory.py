"""
memory.py - Session-based multi-turn conversation memory using LangChain.
Implements InMemoryChatMessageHistory + RunnableWithMessageHistory (Task 8).
"""

from typing import Dict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# In-process session storage mapping session_id -> InMemoryChatMessageHistory
_SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Retrieves or initializes the chat message history for a given session ID.
    """
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]

def clear_session_history(session_id: str) -> None:
    """Clears history for a session ID."""
    if session_id in _SESSION_STORE:
        _SESSION_STORE[session_id].clear()

def format_history_context(history: InMemoryChatMessageHistory) -> str:
    """Formats past messages into a prompt-friendly context string."""
    messages = history.messages
    if not messages:
        return ""
    
    formatted = []
    for msg in messages:
        role = "Patient" if isinstance(msg, HumanMessage) else "Support Agent"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

class ConversationalSessionManager:
    """
    Manages conversational memory across multi-turn interactions.
    Wraps execution inside RunnableWithMessageHistory.
    """
    def __init__(self, execute_fn):
        self.execute_fn = execute_fn
        
        # Runnable pipeline taking {"input": query, "history": history_string}
        def _pipeline(inputs: dict) -> str:
            query = inputs["input"]
            history_context = inputs.get("history", "")
            return self.execute_fn(query, history_context)
            
        runnable = RunnableLambda(_pipeline)
        
        # Wrap with LangChain's RunnableWithMessageHistory
        self.runnable_with_history = RunnableWithMessageHistory(
            runnable,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        
    def chat(self, session_id: str, user_query: str) -> str:
        """
        Runs a chat turn with conversational history tracked under session_id.
        """
        history = get_session_history(session_id)
        history_context = format_history_context(history)
        
        # Execute turn
        response = self.execute_fn(user_query, history_context)
        
        # Record into LangChain message history
        history.add_user_message(user_query)
        history.add_ai_message(response)
        
        return response

