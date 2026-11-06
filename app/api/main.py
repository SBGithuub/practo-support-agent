"""
main.py - FastAPI Application exposing HTTP and WebSocket endpoints for Practo Support Agent.
Implements Tasks 11 and 12.
"""

import os
import time
import uuid
import asyncio
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

from app.agents.schemas import SupportResponse
from app.agents.crew import run_practo_support_pipeline
from app.agents.memory import ConversationalSessionManager
from app.api.logger import log_request_event
from app.rag.chunking import get_fixed_chunks, get_sentence_chunks
from app.rag.vector_store import upsert_chunks
from app.config import COLLECTION_FIXED, COLLECTION_SENTENCE

app = FastAPI(
    title="Practo Domain Support Agent API",
    description="Production-grade AI support backend powered by CrewAI, ChromaDB, and FastAPI.",
    version="1.0.0"
)

# HTTP Request & Response Schemas
class AskRequest(BaseModel):
    query: str = Field(..., description="Patient inquiry or appointment query", min_length=2)
    session_id: Optional[str] = Field(None, description="Optional session ID for multi-turn tracking")

class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., description="Unique policy document identifier, e.g. 'pediatric_guidelines.txt'")
    content: str = Field(..., description="Document content text (2-5 sentences)", min_length=20)
    category: Optional[str] = Field("General Policy", description="Document category or specialty")

class AddDocumentResponse(BaseModel):
    success: bool
    doc_id: str
    chunks_indexed: int
    message: str

# Shared session manager for WebSocket and HTTP stateful chats
def _pipeline_adapter(query: str, history_context: str) -> str:
    res = run_practo_support_pipeline(query, history_context=history_context)
    return res.answer

session_manager = ConversationalSessionManager(_pipeline_adapter)

@app.get("/health")
def health_check():
    """Service health probe."""
    return {"status": "healthy", "service": "practo-support-agent", "mode": "MOCK_LLM"}

@app.post("/ask", response_model=SupportResponse)
def ask_support(request: AskRequest):
    """
    HTTP Endpoint 1: Ask clinic policy or check appointment status.
    Executes guardrails, multi-agent crew, and structured validation.
    """
    start_time = time.perf_counter()
    trace_id = str(uuid.uuid4())
    
    try:
        history_context = ""
        if request.session_id:
            from app.agents.memory import get_session_history, format_history_context
            h = get_session_history(request.session_id)
            history_context = format_history_context(h)
            
        result = run_practo_support_pipeline(request.query, history_context=history_context)
        
        if request.session_id:
            h = get_session_history(request.session_id)
            h.add_user_message(request.query)
            h.add_ai_message(result.answer)
            
        latency = (time.perf_counter() - start_time) * 1000
        log_request_event(
            endpoint="POST /ask",
            raw_query=request.query,
            status="SUCCESS",
            latency_ms=latency,
            metadata={"category": result.category, "confidence": result.confidence, "sources": result.sources},
            trace_id=trace_id
        )
        return result
    except Exception as e:
        latency = (time.perf_counter() - start_time) * 1000
        log_request_event(
            endpoint="POST /ask",
            raw_query=request.query,
            status="ERROR",
            latency_ms=latency,
            metadata={"error": str(e)},
            trace_id=trace_id
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(request: AddDocumentRequest):
    """
    HTTP Endpoint 2: Dynamically add and index a new knowledge base policy document.
    Chunks the document using both strategies and upserts into ChromaDB collections.
    """
    start_time = time.perf_counter()
    trace_id = str(uuid.uuid4())
    
    try:
        doc = {
            "doc_id": request.doc_id,
            "filename": request.doc_id,
            "topic": request.doc_id.replace(".txt", "").replace("_", " ").title(),
            "content": request.content
        }
        
        # 1. Chunk with both strategies
        fixed_chunks = get_fixed_chunks(request.content, request.doc_id)
        sentence_chunks = get_sentence_chunks(request.content, request.doc_id)
        
        # 2. Upsert into respective ChromaDB collections
        upsert_chunks(COLLECTION_FIXED, fixed_chunks)
        upsert_chunks(COLLECTION_SENTENCE, sentence_chunks)
        
        total_chunks = len(fixed_chunks) + len(sentence_chunks)
        latency = (time.perf_counter() - start_time) * 1000
        
        log_request_event(
            endpoint="POST /add-document",
            raw_query=f"Added doc: {request.doc_id}",
            status="SUCCESS",
            latency_ms=latency,
            metadata={"doc_id": request.doc_id, "chunks_indexed": total_chunks},
            trace_id=trace_id
        )
        
        return AddDocumentResponse(
            success=True,
            doc_id=request.doc_id,
            chunks_indexed=total_chunks,
            message=f"Successfully indexed document '{request.doc_id}' into both ChromaDB collections ({total_chunks} total chunks)."
        )
    except Exception as e:
        latency = (time.perf_counter() - start_time) * 1000
        log_request_event(
            endpoint="POST /add-document",
            raw_query=f"Failed adding doc: {request.doc_id}",
            status="ERROR",
            latency_ms=latency,
            metadata={"error": str(e)},
            trace_id=trace_id
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    WebSocket Endpoint: Real-time multi-turn conversational chat.
    Gracefully handles WebSocketDisconnect when client disconnects mid-session without server crashes.
    """
    await websocket.accept()
    session_id = f"ws_{uuid.uuid4().hex[:8]}"
    
    # Send connection greeting
    await websocket.send_json({
        "session_id": session_id,
        "type": "connected",
        "message": "Connected to Practo Support Agent. How can I assist you with clinic policies or appointments today?"
    })
    
    try:
        while True:
            # Await client message
            data = await websocket.receive_text()
            if not data or not data.strip():
                continue
                
            start_time = time.perf_counter()
            trace_id = str(uuid.uuid4())
            
            # Process query through session manager with conversational memory in thread
            response_text = await asyncio.to_thread(session_manager.chat, session_id, data)
            latency = (time.perf_counter() - start_time) * 1000
            
            # Log turn to structured JSON-Lines
            log_request_event(
                endpoint="WS /ws/chat",
                raw_query=data,
                status="SUCCESS",
                latency_ms=latency,
                metadata={"session_id": session_id},
                trace_id=trace_id
            )
            
            # Send response back to WebSocket client
            await websocket.send_json({
                "session_id": session_id,
                "type": "response",
                "answer": response_text
            })
            
    except WebSocketDisconnect:
        # Gracefully handle client disconnect without crashing server
        print(f"[WS] Client {session_id} disconnected gracefully.")
        log_request_event(
            endpoint="WS /ws/chat",
            raw_query=f"Client disconnected: {session_id}",
            status="CLIENT_DISCONNECTED",
            latency_ms=0.0,
            metadata={"session_id": session_id}
        )
    except Exception as e:
        print(f"[WS] Unexpected error for session {session_id}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
