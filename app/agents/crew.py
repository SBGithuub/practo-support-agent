"""
crew.py - CrewAI Multi-Agent Orchestration with Tools, Schema Validation, and Least Autonomy.
Supports Tasks 7, 9, and 15.
"""

import os
import re
from typing import Dict, Any, Optional

# Telemetry must be disabled before CrewAI initialization
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai import Agent, Task, Crew, Process
from app.mock_llm import DeterministicMockLLM
from app.tools.rag_tool import rag_policy_lookup
from app.tools.status_tool import check_appointment_status, check_appointment_status_raw
from app.agents.schemas import SupportResponse
from app.guardrails.pii_masker import mask_pii
from app.guardrails.injection_detector import detect_prompt_injection

def build_practo_crew(query: str, record_id: Optional[str] = None) -> tuple[Crew, Dict[str, Agent]]:
    """
    Constructs the 3-agent Practo Support Crew.
    Enforces the Principle of Least Autonomy (Task 15):
      Only the Lookup Agent is wired with check_appointment_status.
      Only the Retrieval Agent is wired with rag_policy_lookup.
      The Response Composer has no tool privileges.
    """
    mock_llm = DeterministicMockLLM(model="mock-practo-agent")
    
    # 1. Retrieval Agent (Only equipped with rag_policy_lookup)
    retrieval_agent = Agent(
        role="Clinic Policy Retrieval Specialist",
        goal="Search and retrieve verified Practo clinic policies, fees, and operational guidelines.",
        backstory="An expert on Practo's official clinic documentation, scheduling rules, and pricing.",
        tools=[rag_policy_lookup],
        llm=mock_llm,
        verbose=False
    )
    
    # 2. Lookup Agent (Only equipped with check_appointment_status)
    lookup_agent = Agent(
        role="Appointment Record Verification Specialist",
        goal="Lookup appointment statuses, verify consultation charges, and calculate clinical escalation scores.",
        backstory="A clinical records officer with exclusive access to the appointment database.",
        tools=[check_appointment_status],
        llm=mock_llm,
        verbose=False
    )
    
    # 3. Response Composer Agent (Zero tools - Least Autonomy)
    composer_agent = Agent(
        role="Patient Experience Response Composer",
        goal="Synthesize information from specialist agents into an empathetic, clear, grounded response.",
        backstory="A patient relations specialist dedicated to delivering clear, verified answers.",
        tools=[],  # Strictly no tool execution rights
        llm=mock_llm,
        verbose=False
    )
    
    # Enforce Least Autonomy Assertions (Task 15 Application Layer)
    assert check_appointment_status not in retrieval_agent.tools, "Security Violation: Retrieval agent must not access appointment records."
    assert check_appointment_status not in composer_agent.tools, "Security Violation: Composer agent must not access appointment records."
    assert rag_policy_lookup not in lookup_agent.tools, "Security Violation: Lookup agent must not access RAG policies."
    assert rag_policy_lookup not in composer_agent.tools, "Security Violation: Composer agent must not access RAG policies."

    # Determine tasks based on user intent
    tasks = []
    
    if record_id or re.search(r'\b(APT-\d{4})\b', query, re.IGNORECASE):
        # Appointment Lookup Workflow
        rec = record_id or re.search(r'\b(APT-\d{4})\b', query, re.IGNORECASE).group(1).upper()
        task_lookup = Task(
            description=f"Check appointment status and escalation score for patient appointment {rec}. Query: {query}",
            expected_output=f"Detailed status, fee, and escalation details for {rec}.",
            agent=lookup_agent
        )
        task_compose = Task(
            description=f"Compose an empathetic patient response summarizing the verified appointment details for {rec}.",
            expected_output="Final patient response.",
            agent=composer_agent
        )
        tasks = [task_lookup, task_compose]
    else:
        # Knowledge Retrieval Workflow
        task_rag = Task(
            description=f"Retrieve official clinic policies relevant to patient inquiry: '{query}'.",
            expected_output="Retrieved policy context and terms.",
            agent=retrieval_agent
        )
        task_compose = Task(
            description=f"Compose a clear, accurate patient answer for inquiry: '{query}' based on retrieved context.",
            expected_output="Final patient response.",
            agent=composer_agent
        )
        tasks = [task_rag, task_compose]

    crew = Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=False
    )
    
    agents_map = {
        "retrieval": retrieval_agent,
        "lookup": lookup_agent,
        "composer": composer_agent
    }
    return crew, agents_map

def run_practo_support_pipeline(query: str, history_context: str = "") -> SupportResponse:
    """
    Executes the full agent pipeline:
    1. Input Guardrails (PII Masking + Prompt Injection Check)
    2. Context resolution via Session Memory (Task 8)
    3. CrewAI Kickoff with Least Autonomy (Task 7, Task 15)
    4. Structured Pydantic Output Validation (Task 9)
    """
    # Guardrail 1: Prompt Injection Check
    is_injection, reason = detect_prompt_injection(query)
    if is_injection:
        return SupportResponse(
            query=query,
            category="Security Blocked",
            answer="Your request was blocked because it was flagged by our safety filters for potential prompt override.",
            confidence=0.0,
            sources=[],
            action_required=True,
            escalation_score=1.0,
            follow_up_recommended=False
        )
        
    # Guardrail 2: PII Masking
    masked_query, had_pii = mask_pii(query)
    
    # Identify record ID if any in current query or carried from conversation history
    match = re.search(r'\b(APT-\d{4})\b', masked_query, re.IGNORECASE)
    if not match and history_context:
        match = re.search(r'\b(APT-\d{4})\b', history_context, re.IGNORECASE)
    record_id = match.group(1).upper() if match else None
    
    # Build and kickoff crew
    crew, _ = build_practo_crew(masked_query, record_id=record_id)
    raw_result = crew.kickoff()
    raw_text = str(raw_result).strip()
    
    # Structured Output Packaging & Pydantic Validation (Task 9)
    if record_id:
        status_info = check_appointment_status_raw(record_id)
        cat_name = status_info.get("category", "General")
        if not status_info["found"]:
            detailed_ans = f"Appointment record '{record_id}' was not found in the Practo database. Please verify your reference number."
        elif "it" in query.lower() or "follow-up" in query.lower():
            detailed_ans = (
                f"Referencing appointment {record_id} from our prior discussion: "
                f"Status is '{status_info['status']}', Category: {cat_name}, "
                f"Consultation Fee: INR {status_info['consultation_fee_inr']}, "
                f"Follow-up Required: {status_info.get('follow_up_required', False)}, "
                f"Escalation Score: {status_info['escalation_score']} "
                f"({'Escalation Recommended' if status_info['escalation_recommended'] else 'Standard Priority'})."
            )
        else:
            detailed_ans = raw_text
            
        structured = SupportResponse(
            query=masked_query,
            category=f"Appointment {cat_name}",
            answer=detailed_ans,
            confidence=0.95 if status_info["found"] else 0.40,
            sources=[record_id],
            action_required=status_info["escalation_recommended"],
            escalation_score=status_info["escalation_score"],
            follow_up_recommended=status_info.get("follow_up_required", False)
        )
    else:
        # Check if user asks an anaphoric question without any prior appointment context
        if any(w in query.lower().split() for w in ["it", "status", "appointment"]) and not history_context:
            ans_text = "I could not find an appointment reference in our conversation. Please provide your appointment ID (e.g. APT-0001) so I can assist you."
        else:
            ans_text = raw_text
            
        structured = SupportResponse(
            query=masked_query,
            category="Clinic Policy",
            answer=ans_text,
            confidence=0.90,
            sources=["practo_sentence_chunks"],
            action_required=False,
            escalation_score=0.0,
            follow_up_recommended=False
        )
        
    return structured
