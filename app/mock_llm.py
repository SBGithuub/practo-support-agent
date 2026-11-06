"""
mock_llm.py - Deterministic Mock LLM extending crewai.llms.base_llm.BaseLLM.
Requires ZERO API keys and ZERO network access.
Addresses both critical CrewAI pitfalls:
1. Ignores ReAct system prompt template text ("Observation: the result of the action").
2. Dispatches tool calls strictly off declared argument schemas (never tool names).
"""

import os
import re
import json
from typing import Any, List, Dict, Optional, Union, Tuple
from pydantic import BaseModel, Field

# Ensure telemetry is disabled before anything else
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewai.llms.base_llm import BaseLLM
from app.agents.schemas import SupportResponse

class DeterministicMockLLM(BaseLLM):
    """
    Deterministic BaseLLM implementation for offline testing and grading.
    Extends CrewAI's official extension point.
    """
    model: str = "mock-practo-model"
    
    def __init__(self, model: str = "mock-practo-model", **kwargs: Any):
        super().__init__(model=model, **kwargs)
    
    def _extract_user_query_and_observation(self, messages: Any) -> Tuple[str, Optional[str]]:
        """
        Parses messages while strictly ignoring the system prompt template.
        Guards against Pitfall 1: CrewAI's system template containing 'Observation:'
        """
        user_query = ""
        real_observation = None
        
        msg_list = messages if isinstance(messages, list) else [{"role": "user", "content": str(messages)}]
        
        for msg in msg_list:
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
            content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else str(msg))
            
            # CRITICAL: Skip system messages entirely to avoid template false positives
            if role == "system":
                continue
                
            content_str = str(content)
            
            # Check for real user query
            if role == "user" and not user_query:
                # Remove any preamble
                clean_lines = [l for l in content_str.split("\n") if not l.startswith("Observation:") and not l.startswith("Action:")]
                user_query = " ".join(clean_lines).strip()
                
            # Check for real Observation added after tool execution
            # Real observations appear in user/tool return messages
            if "Observation:" in content_str:
                # Extract text after 'Observation:'
                obs_idx = content_str.find("Observation:")
                real_observation = content_str[obs_idx + len("Observation:"):].strip()
                # Stop at next Action or Thought if any
                for stop_token in ["\nThought:", "\nAction:", "\nFinal Answer:"]:
                    if stop_token in real_observation:
                        real_observation = real_observation.split(stop_token)[0].strip()
                        
        return user_query, real_observation

    def call(
        self,
        messages: Union[str, List[Any]],
        tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any
    ) -> str:
        """
        Executes a deterministic step in the agent ReAct loop.
        """
        user_query, observation = self._extract_user_query_and_observation(messages)
        agent_role = getattr(from_agent, "role", "") if from_agent else ""
        
        # 1. If a real tool observation has already been returned:
        if observation:
            final_content = (
                f"Based on the verified lookup results: {observation}. "
                f"Practo ensures patient care and transparent consultation policies."
            )
            return f"Thought: I now have the verified tool result.\nFinal Answer: {final_content}"
            
        # 2. If tools are available for this agent, dispatch using argument schema (Pitfall 2):
        if tools and len(tools) > 0:
            target_tool = None
            action_input = {}
            
            # Search available tools by checking declared argument schemas
            for t in tools:
                tool_name = getattr(t, "name", "")
                schema = getattr(t, "args_schema", None)
                props = {}
                if schema and hasattr(schema, "model_json_schema"):
                    props = schema.model_json_schema().get("properties", {})
                elif schema and hasattr(schema, "schema"):
                    props = schema.schema().get("properties", {})
                elif hasattr(t, "description"):
                    pass
                    
                # Schema Dispatch Rule A: Tool requiring 'record_id'
                if "record_id" in props or "record_id" in str(schema):
                    match = re.search(r'\b(APT-\d{4})\b', str(messages), re.IGNORECASE)
                    record_id = match.group(1).upper() if match else "APT-0001"
                    target_tool = tool_name or "check_appointment_status"
                    action_input = {"record_id": record_id}
                    break
                    
                # Schema Dispatch Rule B: Tool requiring 'query' or 'query_text'
                if "query" in props or "query_text" in props or "query" in str(schema):
                    # Clean out agent prompts from query
                    match_apt = re.search(r'record_id[:\s]+(APT-\d{4})', str(messages), re.IGNORECASE)
                    clean_query = user_query or "appointment policy"
                    target_tool = tool_name or "rag_policy_lookup"
                    action_input = {"query": clean_query[:100]}
                    break

            if target_tool:
                return (
                    f"Thought: I need to consult the verified clinical tool for accurate information.\n"
                    f"Action: {target_tool}\n"
                    f"Action Input: {json.dumps(action_input)}"
                )

        # 3. If no tools are assigned (e.g. Response Composer Agent) or tool dispatch not needed:
        # Generate composed final answer
        match = re.search(r'\b(APT-\d{4})\b', str(messages), re.IGNORECASE)
        apt_str = match.group(1).upper() if match else None
        
        if apt_str:
            final_text = (
                f"Appointment {apt_str} has been verified in the Practo clinical records database. "
                f"The consultation status and specialty details have been confirmed. "
                f"Please refer to standard clinic guidelines for arrival and check-in procedures."
            )
        else:
            final_text = (
                "Practo clinics operate under standard transparent healthcare policies. "
                "Cancellations up to two hours prior are penalty-free, and follow-up visits within seven days receive courtesy discounts. "
                "For emergencies, patients should immediately contact emergency hospital triage at 108/112."
            )
            
        return f"Thought: I will compose the final synthesized patient response.\nFinal Answer: {final_text}"

    async def acall(
        self,
        messages: Union[str, List[Any]],
        tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        response_model: Optional[type[BaseModel]] = None,
        **kwargs: Any
    ) -> str:
        """
        Asynchronous call delegating to synchronous deterministic implementation.
        Required by CrewAI 1.15+ AgentExecutor kickoff_async flow.
        """
        return self.call(
            messages=messages,
            tools=tools,
            callbacks=callbacks,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
            **kwargs
        )
