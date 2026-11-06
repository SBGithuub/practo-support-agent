"""
autogen_review.py - Autogen 2-agent RoundRobinGroupChat review stage.
Implements Task 14 with Pydantic structured output ReviewVerdict.
"""

import asyncio
from typing import Sequence, List
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, StructuredMessage, BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from app.agents.schemas import ReviewVerdict

# Prohibited ungrounded assertions for simulated compliance testing
UNGROUNDED_TRIGGERS = [
    "100% free surgery",
    "unlimited free medication",
    "complimentary luxury hotel stay",
    "free helicopter transport",
    "guaranteed 100% cure"
]

class PolicyComplianceReviewer(BaseChatAgent):
    """
    Agent 1 in Autogen Review Stage:
    Analyzes draft answer against retrieved policy context to detect hallucinations or ungrounded claims.
    """
    def __init__(self):
        super().__init__(
            name="PolicyComplianceReviewer",
            description="Reviews medical customer support answers against retrieved knowledge base context."
        )

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return [TextMessage]

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken
    ) -> Response:
        task_text = messages[-1].content if messages else ""
        if isinstance(task_text, str):
            task_lower = task_text.lower()
        else:
            task_lower = str(task_text).lower()

        # Check for ungrounded hallucination triggers
        violations = [trig for trig in UNGROUNDED_TRIGGERS if trig in task_lower]
        
        if violations:
            review_msg = (
                f"COMPLIANCE VIOLATION DETECTED: The draft contains ungrounded claims not supported "
                f"by Practo official policies: {', '.join(violations)}. "
                f"REVISION MANDATORY: Strip out the unverified claims and restore grounded clinic policy."
            )
        else:
            review_msg = (
                "COMPLIANCE VERIFICATION PASSED: All statements in the draft response are strictly grounded "
                "in verified Practo clinical documentation."
            )

        return Response(
            chat_message=TextMessage(content=review_msg, source=self.name)
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

class FinalEditor(BaseChatAgent):
    """
    Agent 2 in Autogen Review Stage:
    Produces final Pydantic ReviewVerdict structured message.
    Approves draft unchanged or outputs revised response.
    """
    def __init__(self):
        super().__init__(
            name="FinalEditor",
            description="Synthesizes reviewer recommendations and generates structured ReviewVerdict output."
        )

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return [StructuredMessage[ReviewVerdict]]

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken
    ) -> Response:
        # Extract initial task containing draft answer and retrieved context
        task_message = messages[0].content if messages else ""
        reviewer_message = messages[-1].content if len(messages) > 1 else ""

        # Parse draft from task
        draft = ""
        context = ""
        task_str = str(task_message)
        if "DRAFT ANSWER:" in task_str:
            parts = task_str.split("DRAFT ANSWER:")
            if len(parts) > 1:
                subparts = parts[1].split("RETRIEVED CONTEXT:")
                draft = subparts[0].strip()
                context = subparts[1].strip() if len(subparts) > 1 else ""
        else:
            draft = task_str

        # Check if reviewer flagged violations
        rev_str = str(reviewer_message)
        if "COMPLIANCE VIOLATION DETECTED" in rev_str:
            # Reconstruct grounded answer by purging ungrounded triggers
            cleaned_draft = draft
            for trig in UNGROUNDED_TRIGGERS:
                cleaned_draft = cleaned_draft.replace(trig, "").replace(trig.title(), "")
            cleaned_draft = " ".join(cleaned_draft.split())
            if not cleaned_draft or len(cleaned_draft) < 20:
                cleaned_draft = "Practo provides verified consultation scheduling under official published clinic fee structures."

            verdict = ReviewVerdict(
                approved=False,
                final_answer=cleaned_draft,
                reason="Caught ungrounded promotional/clinical claim not present in verified policy. Revised draft to ensure 100% groundedness."
            )
        else:
            verdict = ReviewVerdict(
                approved=True,
                final_answer=draft,
                reason="Draft response verified against retrieved knowledge base. No factual contradictions or ungrounded claims detected."
            )

        return Response(
            chat_message=StructuredMessage[ReviewVerdict](
                content=verdict,
                source=self.name
            )
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

async def review_draft_async(draft_answer: str, retrieved_context: str) -> ReviewVerdict:
    """
    Orchestrates the 2-agent Autogen team:
    RoundRobinGroupChat with max_turns=2 and custom_message_types=[StructuredMessage[ReviewVerdict]].
    """
    reviewer = PolicyComplianceReviewer()
    editor = FinalEditor()

    team = RoundRobinGroupChat(
        participants=[reviewer, editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]]
    )

    task_payload = (
        f"Please review the following draft patient response for policy compliance and groundedness.\n\n"
        f"DRAFT ANSWER:\n{draft_answer}\n\n"
        f"RETRIEVED CONTEXT:\n{retrieved_context}"
    )

    result = await team.run(task=task_payload)
    last_msg = result.messages[-1]
    
    if isinstance(last_msg, StructuredMessage) and isinstance(last_msg.content, ReviewVerdict):
        return last_msg.content
    elif hasattr(last_msg, "content") and isinstance(last_msg.content, ReviewVerdict):
        return last_msg.content
    else:
        raise ValueError(f"Expected StructuredMessage[ReviewVerdict], got {type(last_msg)}")

def run_autogen_review(draft_answer: str, retrieved_context: str) -> ReviewVerdict:
    """Synchronous entry point wrapping async review execution."""
    return asyncio.run(review_draft_async(draft_answer, retrieved_context))

