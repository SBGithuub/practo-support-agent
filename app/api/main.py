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
from fastapi.responses import HTMLResponse
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

def get_browser_chat_html() -> str:
    """Returns self-contained responsive HTML for real-time WebSocket chat in browser."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Practo Support Agent | WebSocket Live Chat</title>
  <style>
    :root {
      --primary: #28328c;
      --primary-dark: #1b2559;
      --accent: #14bef0;
      --bg: #f4f6fa;
      --card-bg: #ffffff;
      --text: #1d2129;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --success: #10b981;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh; }
    header { background: var(--primary-dark); color: #fff; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .logo-container { display: flex; align-items: center; gap: 12px; }
    .logo-badge { background: var(--accent); color: #fff; font-weight: 800; padding: 5px 10px; border-radius: 6px; font-size: 13px; letter-spacing: 0.5px; }
    .title-group h1 { font-size: 17px; font-weight: 600; }
    .title-group p { font-size: 11px; color: #94a3b8; }
    .status-group { display: flex; align-items: center; gap: 10px; }
    .status-badge { display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 500; padding: 6px 12px; border-radius: 20px; background: rgba(255,255,255,0.1); }
    .status-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--danger); transition: all 0.3s; }
    .status-dot.connected { background: var(--success); box-shadow: 0 0 8px var(--success); }
    .btn-action { padding: 5px 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.3); background: transparent; color: #fff; font-size: 12px; cursor: pointer; transition: all 0.2s; }
    .btn-action:hover { background: rgba(255,255,255,0.2); }
    main { flex: 1; display: flex; flex-direction: column; max-width: 960px; width: 100%; margin: 0 auto; padding: 16px; overflow: hidden; }
    .suggestions { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }
    .suggestions span { font-size: 12px; color: var(--text-muted); font-weight: 500; }
    .chip { background: #fff; border: 1px solid var(--border); padding: 6px 12px; border-radius: 16px; font-size: 12px; color: var(--primary); cursor: pointer; transition: all 0.2s; }
    .chip:hover { background: var(--primary); color: #fff; border-color: var(--primary); transform: translateY(-1px); }
    .chat-window { flex: 1; background: var(--card-bg); border-radius: 12px; border: 1px solid var(--border); padding: 18px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
    .message { display: flex; flex-direction: column; max-width: 78%; line-height: 1.5; font-size: 14px; }
    .message.user { align-self: flex-end; }
    .message.agent { align-self: flex-start; }
    .message.system { align-self: center; max-width: 90%; font-size: 12px; color: var(--text-muted); background: #f1f5f9; padding: 6px 12px; border-radius: 12px; text-align: center; }
    .bubble { padding: 12px 16px; border-radius: 12px; word-break: break-word; }
    .message.user .bubble { background: var(--primary); color: #fff; border-bottom-right-radius: 2px; }
    .message.agent .bubble { background: #f8fafc; border: 1px solid var(--border); color: var(--text); border-bottom-left-radius: 2px; }
    .msg-meta { font-size: 11px; margin-top: 4px; color: var(--text-muted); }
    .message.user .msg-meta { text-align: right; }
    .typing-indicator { display: none; align-self: flex-start; background: #f8fafc; border: 1px solid var(--border); padding: 10px 14px; border-radius: 12px; font-size: 12px; color: var(--text-muted); }
    .input-bar { margin-top: 12px; display: flex; gap: 10px; }
    .input-bar input { flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid var(--border); font-size: 14px; outline: none; transition: border-color 0.2s; }
    .input-bar input:focus { border-color: var(--primary); }
    .input-bar button { padding: 12px 24px; border-radius: 8px; border: none; background: var(--primary); color: #fff; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    .input-bar button:hover { background: var(--primary-dark); }
    .input-bar button:disabled { background: #cbd5e1; cursor: not-allowed; }
  </style>
</head>
<body>
  <header>
    <div class="logo-container">
      <div class="logo-badge">practo</div>
      <div class="title-group">
        <h1>Practo Support Agent</h1>
        <p>Live WebSocket Conversational Session (Task 11)</p>
      </div>
    </div>
    <div class="status-group">
      <div class="status-badge">
        <span class="status-dot" id="statusDot"></span>
        <span id="statusText">Connecting...</span>
      </div>
      <button class="btn-action" id="toggleConnectBtn" onclick="toggleConnection()">Disconnect</button>
    </div>
  </header>

  <main>
    <div class="suggestions">
      <span>Quick Inquiries:</span>
      <button class="chip" onclick="sendQuickPrompt('What is the cancellation window for appointments?')">Cancellation Policy</button>
      <button class="chip" onclick="sendQuickPrompt('Check status for appointment APT-0007')">Check APT-0007</button>
      <button class="chip" onclick="sendQuickPrompt('Is follow-up required for it, and what is its escalation score?')">Follow-up for it? (Memory)</button>
      <button class="chip" onclick="sendQuickPrompt('What is the emergency visit protocol if I have sudden chest pain?')">Emergency Protocol</button>
      <button class="chip" onclick="sendQuickPrompt('My contact is +91 9876543210. Check APT-0001 status')">PII Masking Test</button>
    </div>

    <div class="chat-window" id="chatWindow">
      <!-- Messages appear here -->
    </div>

    <div class="typing-indicator" id="typingIndicator">
      Practo Agent is formulating verified response...
    </div>

    <form class="input-bar" id="chatForm" onsubmit="handleSend(event)">
      <input type="text" id="userInput" placeholder="Type a clinic policy question or appointment ID (e.g. APT-0001)..." autocomplete="off" />
      <button type="submit" id="sendBtn">Send</button>
    </form>
  </main>

  <script>
    let ws = null;
    let isExplicitlyClosed = false;
    const chatWindow = document.getElementById("chatWindow");
    const userInput = document.getElementById("userInput");
    const sendBtn = document.getElementById("sendBtn");
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");
    const toggleConnectBtn = document.getElementById("toggleConnectBtn");
    const typingIndicator = document.getElementById("typingIndicator");

    function connectWebSocket() {
      isExplicitlyClosed = false;
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${protocol}//${window.location.host}/ws/chat`;

      statusDot.className = "status-dot";
      statusText.textContent = "Connecting...";
      toggleConnectBtn.textContent = "Disconnect";

      ws = new WebSocket(wsUrl);

      ws.onopen = function() {
        statusDot.className = "status-dot connected";
        statusText.textContent = "Connected (ws/chat)";
        sendBtn.disabled = false;
      };

      ws.onmessage = function(event) {
        typingIndicator.style.display = "none";
        try {
          const data = JSON.parse(event.data);
          if (data.type === "connected") {
            appendSystemMessage(data.message + " (Session ID: " + data.session_id + ")");
          } else if (data.type === "response") {
            appendAgentMessage(data.answer);
          }
        } catch (e) {
          appendAgentMessage(event.data);
        }
      };

      ws.onclose = function() {
        statusDot.className = "status-dot";
        statusText.textContent = "Disconnected";
        sendBtn.disabled = true;
        toggleConnectBtn.textContent = "Reconnect";
        typingIndicator.style.display = "none";
        if (!isExplicitlyClosed) {
          appendSystemMessage("Connection closed. Click 'Reconnect' to resume session.");
        }
      };

      ws.onerror = function() {
        statusDot.className = "status-dot";
        statusText.textContent = "Connection Error";
        typingIndicator.style.display = "none";
      };
    }

    function toggleConnection() {
      if (ws && ws.readyState === WebSocket.OPEN) {
        isExplicitlyClosed = true;
        ws.close();
        appendSystemMessage("Client intentionally disconnected mid-conversation. Server remains healthy (Task 11).");
      } else {
        connectWebSocket();
      }
    }

    function appendUserMessage(text) {
      const msgDiv = document.createElement("div");
      msgDiv.className = "message user";
      const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      msgDiv.innerHTML = `<div class="bubble">${escapeHtml(text)}</div><div class="msg-meta">Patient · ${timeStr}</div>`;
      chatWindow.appendChild(msgDiv);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function appendAgentMessage(text) {
      const msgDiv = document.createElement("div");
      msgDiv.className = "message agent";
      const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      msgDiv.innerHTML = `<div class="bubble">${escapeHtml(text)}</div><div class="msg-meta">Practo Support Agent · ${timeStr}</div>`;
      chatWindow.appendChild(msgDiv);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function appendSystemMessage(text) {
      const msgDiv = document.createElement("div");
      msgDiv.className = "message system";
      msgDiv.textContent = text;
      chatWindow.appendChild(msgDiv);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function escapeHtml(string) {
      const entityMap = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
      return String(string).replace(/[&<>"']/g, function (s) { return entityMap[s]; });
    }

    function handleSend(event) {
      event.preventDefault();
      const text = userInput.value.trim();
      if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

      appendUserMessage(text);
      ws.send(text);
      userInput.value = "";
      typingIndicator.style.display = "block";
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    function sendQuickPrompt(promptText) {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket();
        setTimeout(() => {
          if (ws && ws.readyState === WebSocket.OPEN) {
            userInput.value = promptText;
            document.getElementById("chatForm").dispatchEvent(new Event("submit"));
          }
        }, 500);
        return;
      }
      userInput.value = promptText;
      document.getElementById("chatForm").dispatchEvent(new Event("submit"));
    }

    // Auto-connect on load
    window.addEventListener("DOMContentLoaded", connectWebSocket);
  </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
def browser_chat_ui():
    """Serves the interactive browser-based WebSocket chat client."""
    return HTMLResponse(content=get_browser_chat_html())

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
