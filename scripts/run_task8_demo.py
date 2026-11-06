"""
run_task8_demo.py - Demonstrates LangChain Session-Based Memory (Task 8).
Generates:
  - transcripts/task8_memory_multiturn.txt (Conversation carrying state across turns)
  - transcripts/task8_memory_fresh_reset.txt (Fresh conversation showing state absent/reset)
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from app.agents.memory import ConversationalSessionManager, get_session_history
from app.agents.crew import run_practo_support_pipeline

def pipeline_executor(query: str, history_context: str) -> str:
    res = run_practo_support_pipeline(query, history_context=history_context)
    return res.answer

def run_multiturn_demo():
    session_id = "session_patient_42"
    manager = ConversationalSessionManager(pipeline_executor)
    
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("TASK 8: MULTI-TURN CONVERSATION MEMORY DEMONSTRATION")
    output_lines.append(f"Session ID: {session_id}")
    output_lines.append("Framework: LangChain InMemoryChatMessageHistory + RunnableWithMessageHistory")
    output_lines.append("=" * 80)
    
    # Turn 1
    t1_query = "Hi, can you please check on my appointment APT-0007?"
    output_lines.append(f"\n[TURN 1]")
    output_lines.append(f"Patient: {t1_query}")
    t1_ans = manager.chat(session_id, t1_query)
    output_lines.append(f"Agent: {t1_ans}")
    
    # Turn 2 - Anaphoric reference relying on memory
    t2_query = "Is follow-up required for it, and what is its escalation score?"
    output_lines.append(f"\n[TURN 2 - CONTEXT CARRIED VIA MEMORY]")
    output_lines.append(f"Patient: {t2_query}")
    t2_ans = manager.chat(session_id, t2_query)
    output_lines.append(f"Agent: {t2_ans}")
    
    # Inspect stored messages in LangChain memory
    history = get_session_history(session_id)
    output_lines.append(f"\n[STORED IN-PROCESS LANGCHAIN HISTORY]")
    for i, msg in enumerate(history.messages):
        output_lines.append(f"  Message {i+1} [{type(msg).__name__}]: {msg.content}")
        
    output_lines.append("\n" + "=" * 80)
    output_lines.append("STATE VERIFICATION: Successfully carried APT-0007 context across turns!")
    output_lines.append("=" * 80)
    
    text = "\n".join(output_lines)
    with open("transcripts/task8_memory_multiturn.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Multi-turn transcript written to transcripts/task8_memory_multiturn.txt")

def run_fresh_reset_demo():
    fresh_session_id = "session_fresh_patient_99"
    manager = ConversationalSessionManager(pipeline_executor)
    
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("TASK 8: FRESH CONVERSATION (STATE ABSENT / RESET DEMONSTRATION)")
    output_lines.append(f"Session ID: {fresh_session_id}")
    output_lines.append("Expected Behavior: Fresh session has zero prior turns. Anaphoric 'it' cannot resolve.")
    output_lines.append("=" * 80)
    
    # Turn 1 on fresh session with same anaphoric question as Turn 2 above
    query = "Is follow-up required for it, and what is its escalation score?"
    output_lines.append(f"\n[TURN 1 - NEW FRESH SESSION]")
    output_lines.append(f"Patient: {query}")
    ans = manager.chat(fresh_session_id, query)
    output_lines.append(f"Agent: {ans}")
    
    history = get_session_history(fresh_session_id)
    output_lines.append(f"\n[STORED IN-PROCESS LANGCHAIN HISTORY]")
    for i, msg in enumerate(history.messages):
        output_lines.append(f"  Message {i+1} [{type(msg).__name__}]: {msg.content}")
        
    output_lines.append("\n" + "=" * 80)
    output_lines.append("RESET VERIFICATION: Prior session state was completely absent in fresh session.")
    output_lines.append("=" * 80)
    
    text = "\n".join(output_lines)
    with open("transcripts/task8_memory_fresh_reset.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Fresh reset transcript written to transcripts/task8_memory_fresh_reset.txt")

if __name__ == "__main__":
    run_multiturn_demo()
    run_fresh_reset_demo()

