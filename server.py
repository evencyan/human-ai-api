"""
AI API Server
=============
OpenAI-compatible API server.

Supports: text responses, function/tool calls, tool_choice

Usage:
    python server.py [--host HOST] [--port PORT] [--timeout SECONDS]
"""

import asyncio
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ============================================================================
# Configuration
# ============================================================================

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8001"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "300"))

# Runtime settings (modifiable via admin API)
runtime_settings = {
    "timeout": REQUEST_TIMEOUT,
}

# ============================================================================
# App
# ============================================================================

app = FastAPI(
    title="AI API",
    description="OpenAI-compatible API",
    version="1.1.0",
)

# Allow all origins — required for browser-based clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# ============================================================================
# Storage
# ============================================================================

pending_requests: Dict[str, Dict[str, Any]] = {}
completed_requests: List[Dict[str, Any]] = []
MAX_HISTORY = 500

# Thread tracking — group related requests into conversations
threads: Dict[str, Dict[str, Any]] = {}
thread_expiry = 3600  # 1 hour


def _get_thread_id(messages: List[Any]) -> str:
    """Generate a thread ID from the conversation's first user message."""
    first_user = None
    for m in messages:
        if isinstance(m, dict):
            if m.get("role") == "user":
                first_user = str(m.get("content", ""))[:100]
                break
        elif hasattr(m, "role") and m.role == "user":
            first_user = str(getattr(m, "content", ""))[:100]
            break
    return hashlib.md5((first_user or str(time.time())).encode()).hexdigest()[:12]


# ============================================================================
# OpenAI-compatible Models
# ============================================================================

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    model_config = {"extra": "allow"}


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Any] = None

    model_config = {"extra": "allow"}


class ToolCallInput(BaseModel):
    """A tool call that the human operator submits."""
    name: str
    arguments: str  # JSON string


class RespondBody(BaseModel):
    request_id: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallInput]] = None


# ============================================================================
# Anthropic-compatible Models
# ============================================================================

class AnthropicContent(BaseModel):
    type: str
    text: Optional[str] = None
    # tool_use fields
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    # tool_result fields
    tool_use_id: Optional[str] = None
    content: Optional[Any] = None
    is_error: Optional[bool] = None

    model_config = {"extra": "allow"}


class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class AnthropicMessage(BaseModel):
    role: str
    content: Any  # string or list of content blocks

    model_config = {"extra": "allow"}


class AnthropicRequest(BaseModel):
    model: str = "claude-4"
    messages: List[AnthropicMessage]
    max_tokens: int = 1024
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    system: Optional[Any] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Any] = None

    model_config = {"extra": "allow"}


def _anthropic_to_internal(areq: AnthropicRequest) -> ChatCompletionRequest:
    """Convert Anthropic request to internal OpenAI-compatible format."""
    messages = []
    # Add system prompt as a message
    if areq.system:
        sys_text = areq.system if isinstance(areq.system, str) else (
            areq.system[0].get("text", "") if isinstance(areq.system, list) and areq.system else ""
        )
        if sys_text:
            messages.append(Message(role="system", content=sys_text))
    # Convert messages
    for m in areq.messages:
        if isinstance(m.content, str):
            messages.append(Message(role=m.role, content=m.content))
        elif isinstance(m.content, list):
            parts = []
            tool_calls = []
            for block in m.content:
                if isinstance(block, dict):
                    t = block.get("type")
                elif hasattr(block, "type"):
                    t = block.type
                else:
                    continue
                if t == "text":
                    parts.append(block.get("text", "") if isinstance(block, dict) else block.text)
                elif t == "tool_use":
                    tc_id = block.get("id", "") if isinstance(block, dict) else block.id
                    tc_name = block.get("name", "") if isinstance(block, dict) else block.name
                    tc_input = block.get("input", {}) if isinstance(block, dict) else block.input
                    tool_calls.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": tc_name, "arguments": json.dumps(tc_input, ensure_ascii=False)},
                    })
                elif t == "tool_result":
                    content_str = block.get("content", "") if isinstance(block, dict) else block.content
                    tool_use_id = block.get("tool_use_id", "") if isinstance(block, dict) else block.tool_use_id
                    messages.append(Message(role="tool", content=str(content_str) if not isinstance(content_str, str) else content_str, tool_call_id=tool_use_id))
            if tool_calls:
                messages.append(Message(role="assistant", content="\n".join(parts) if parts else None, tool_calls=tool_calls))
            elif parts:
                messages.append(Message(role=m.role, content="\n".join(parts)))
    # Convert tools
    tools = None
    if areq.tools:
        tools = []
        for t in areq.tools:
            if isinstance(t, dict):
                tools.append(Tool(type="function", function=ToolFunction(
                    name=t.get("name", ""),
                    description=t.get("description"),
                    parameters=t.get("input_schema"),
                )))
            else:
                tools.append(Tool(type="function", function=ToolFunction(
                    name=t.name,
                    description=t.description,
                    parameters=t.input_schema,
                )))
    # Convert tool_choice
    tool_choice = None
    if areq.tool_choice:
        if isinstance(areq.tool_choice, str):
            if areq.tool_choice == "any":
                tool_choice = "required"
            else:
                tool_choice = areq.tool_choice
        elif isinstance(areq.tool_choice, dict):
            tc_type = areq.tool_choice.get("type")
            if tc_type == "any":
                tool_choice = "required"
            elif tc_type == "tool":
                tn = areq.tool_choice.get("name")
                if tn:
                    tool_choice = {"type": "function", "function": {"name": tn}}
            elif tc_type == "function":
                tool_choice = areq.tool_choice
    return ChatCompletionRequest(
        model=areq.model,
        messages=messages,
        max_tokens=areq.max_tokens,
        stream=areq.stream,
        temperature=areq.temperature,
        tools=tools,
        tool_choice=tool_choice,
    )


def _internal_to_anthropic(request_id: str, model: str, content: Optional[str], tool_calls: Optional[List[Dict]]) -> dict:
    """Convert internal response to Anthropic format."""
    # Build stop_reason, output content blocks, and usage
    input_tokens = 0  # estimated by caller
    if tool_calls:
        blocks = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            try:
                parsed = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                parsed = {}
            blocks.append({"type": "tool_use", "id": tc.get("id", ""), "name": fn.get("name", ""), "input": parsed})
        output_text = json.dumps(blocks)
        stop_reason = "tool_use"
    else:
        blocks = [{"type": "text", "text": content or ""}]
        output_text = content or ""
        stop_reason = "end_turn"
    output_tokens = _estimate_tokens_text(output_text)
    return {
        "id": request_id.replace("chatcmpl-", "msg_"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


# ============================================================================
# OpenAI-compatible API Endpoints
# ============================================================================

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible format)."""
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4", "object": "model", "created": 1700000000, "owned_by": "openai", "context_window": 131072},
            {"id": "gpt-4o", "object": "model", "created": 1700000000, "owned_by": "openai", "context_window": 131072},
            {"id": "claude-4", "object": "model", "created": 1700000000, "owned_by": "anthropic", "context_window": 200000},
        ],
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(req: ChatCompletionRequest):
    """
    Create a chat completion — OpenAI-compatible.
    Blocks until an operator responds via the admin dashboard.
    Supports both text and tool_call responses.
    """
    # Streaming is silently ignored — we always return full response
    # (most clients handle this gracefully)

    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_at = time.time()

    # Serialise tools / tool_choice for the admin panel
    tools_raw = None
    if req.tools:
        tools_raw = [t.model_dump() for t in req.tools]
    tool_choice_raw = None
    if req.tool_choice is not None:
        if isinstance(req.tool_choice, str):
            tool_choice_raw = req.tool_choice
        elif hasattr(req.tool_choice, 'model_dump'):
            tool_choice_raw = req.tool_choice.model_dump()
        else:
            tool_choice_raw = req.tool_choice

    response_event = asyncio.Event()
    response_data: Dict[str, Any] = {
        "content": None,
        "tool_calls": None,
        "error": None,
    }

    thread_id = _get_thread_id(req.messages)

    pending_requests[request_id] = {
        "id": request_id,
        "thread_id": thread_id,
        "model": req.model,
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "user": req.user,
        "tools": tools_raw,
        "tool_choice": tool_choice_raw,
        "created_at": created_at,
        "status": "pending",
        "response_event": response_event,
        "response_data": response_data,
    }

    # ---- wait for operator response ----
    try:
        await asyncio.wait_for(response_event.wait(), timeout=runtime_settings["timeout"] if runtime_settings["timeout"] > 0 else None)
    except asyncio.TimeoutError:
        if request_id in pending_requests:
            pending_requests[request_id]["status"] = "timeout"
            _archive_request(request_id)
        timeout_val = runtime_settings["timeout"]
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {timeout_val}s.",
        )

    if response_data.get("error"):
        _archive_request(request_id)
        raise HTTPException(status_code=500, detail=response_data["error"])

    content = response_data.get("content")
    tool_calls = response_data.get("tool_calls")

    _archive_request(request_id, response=content, tool_calls=tool_calls)

    now = int(time.time())

    if req.stream:
        # SSE streaming response
        return StreamingResponse(
            _stream_response(request_id, now, req.model, content, tool_calls),
            media_type="text/event-stream",
        )

    # Non-streaming response
    if tool_calls:
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": now,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": _estimate_tokens(req.messages),
                "completion_tokens": _estimate_tool_tokens(tool_calls),
                "total_tokens": _estimate_tokens(req.messages) + _estimate_tool_tokens(tool_calls),
            },
        }
    else:
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": now,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": _estimate_tokens(req.messages),
                "completion_tokens": _estimate_tokens_text(content),
                "total_tokens": _estimate_tokens(req.messages) + _estimate_tokens_text(content),
            },
        }


# ============================================================================
# Anthropic-compatible Endpoint
# ============================================================================

@app.post("/v1/messages")
async def create_message(areq: AnthropicRequest):
    """Anthropic Messages API — converted to internal format internally."""
    # Convert to internal format
    ireq = _anthropic_to_internal(areq)
    stream = areq.stream or False

    request_id = f"msg_{uuid.uuid4().hex[:24]}"
    created_at = time.time()

    response_event = asyncio.Event()
    response_data: Dict[str, Any] = {"content": None, "tool_calls": None, "error": None}

    # Serialise tools for admin panel (preserve Anthropic format)
    tools_raw = None
    if areq.tools:
        tools_raw = [t.model_dump() if hasattr(t, 'model_dump') else t for t in areq.tools]
    tool_choice_raw = areq.tool_choice

    thread_id = _get_thread_id(ireq.messages)

    pending_requests[request_id] = {
        "id": request_id,
        "thread_id": thread_id,
        "model": areq.model,
        "messages": [m.model_dump() for m in ireq.messages],
        "temperature": areq.temperature,
        "max_tokens": areq.max_tokens,
        "user": None,
        "tools": tools_raw,
        "tool_choice": tool_choice_raw,
        "created_at": created_at,
        "status": "pending",
        "response_event": response_event,
        "response_data": response_data,
    }

    try:
        await asyncio.wait_for(response_event.wait(), timeout=runtime_settings["timeout"] if runtime_settings["timeout"] > 0 else None)
    except asyncio.TimeoutError:
        if request_id in pending_requests:
            pending_requests[request_id]["status"] = "timeout"
            _archive_request(request_id)
        timeout_val = runtime_settings["timeout"]
        raise HTTPException(status_code=504, detail=f"Request timed out after {timeout_val}s.")

    if response_data.get("error"):
        _archive_request(request_id)
        raise HTTPException(status_code=500, detail=response_data["error"])

    content = response_data.get("content")
    tool_calls = response_data.get("tool_calls")

    _archive_request(request_id, response=content, tool_calls=tool_calls)

    if stream:
        return StreamingResponse(
            _anthropic_stream(request_id, areq.model, content, tool_calls),
            media_type="text/event-stream",
        )

    result = _internal_to_anthropic(request_id, areq.model, content, tool_calls)
    # Estimate input tokens
    input_tokens = _estimate_tokens(ireq.messages)
    result["usage"]["input_tokens"] = input_tokens
    return result


# ============================================================================
# Admin API Endpoints
# ============================================================================

@app.get("/admin/api/requests")
async def admin_get_requests():
    """Return pending requests + recent history for the dashboard."""
    now = time.time()

    pending = []
    for rid, r in pending_requests.items():
        pending.append({
            "id": rid,
            "model": r["model"],
            "messages": r["messages"],
            "temperature": r.get("temperature"),
            "max_tokens": r.get("max_tokens"),
            "user": r.get("user"),
            "tools": r.get("tools"),
            "tool_choice": r.get("tool_choice"),
            "created_at": r["created_at"],
            "status": r["status"],
            "wait_seconds": round(now - r["created_at"], 1),
        })
    pending.sort(key=lambda x: x["created_at"])

    recent = []
    for r in completed_requests[-50:]:
        recent.append({
            "id": r["id"],
            "model": r["model"],
            "messages": r.get("messages", []),
            "tools": r.get("tools"),
            "tool_choice": r.get("tool_choice"),
            "created_at": r.get("created_at"),
            "completed_at": r.get("completed_at"),
            "status": r.get("status", "completed"),
            "response": r.get("response"),
            "tool_calls": r.get("tool_calls"),
            "wait_seconds": (
                round((r.get("completed_at", 0) - r.get("created_at", 0)), 1)
                if r.get("completed_at") and r.get("created_at")
                else None
            ),
        })
    recent.reverse()

    return {"pending": pending, "recent": recent, "server_time": now}


@app.post("/admin/api/respond")
async def admin_respond(body: RespondBody):
    """Submit a human response (text or tool call) to a pending request."""
    if body.request_id not in pending_requests:
        raise HTTPException(status_code=404, detail="Request not found or already handled")

    if not body.content and not body.tool_calls:
        raise HTTPException(status_code=400, detail="Must provide content or tool_calls")

    req = pending_requests[body.request_id]

    # If tool_choice is required, force tool call response
    tc = req.get("tool_choice")
    # tool_choice=required or specific tool → must respond with tool call
    # tool_choice=any/auto or unset → text response is fine
    tc_is_required = tc == "required" or (isinstance(tc, dict) and tc.get("type") in ("tool", "function"))
    if tc_is_required and not body.tool_calls:
        raise HTTPException(
            status_code=400,
            detail="tool_choice=required: you MUST respond with a tool call, not text",
        )

    if body.tool_calls:
        # Build OpenAI-compatible tool_call objects with generated IDs
        tool_calls = []
        for tc in body.tool_calls:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            })
        req["response_data"]["tool_calls"] = tool_calls
        req["response_data"]["content"] = None
    else:
        req["response_data"]["content"] = body.content.strip()
        req["response_data"]["tool_calls"] = None

    req["response_event"].set()
    return {"status": "ok", "request_id": body.request_id}


@app.post("/admin/api/skip")
async def admin_skip(request_id: str = Query(...)):
    """Skip/reject a pending request."""
    if request_id not in pending_requests:
        raise HTTPException(status_code=404, detail="Request not found or already handled")

    req = pending_requests[request_id]
    req["response_data"]["error"] = "Request could not be processed"
    req["response_event"].set()
    return {"status": "ok", "request_id": request_id}


@app.get("/admin/api/stats")
async def admin_stats():
    """Server statistics."""
    return {
        "pending_count": len(pending_requests),
        "completed_count": len(completed_requests),
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "timeout": runtime_settings["timeout"],
    }


class SettingsUpdate(BaseModel):
    timeout: Optional[int] = None


@app.get("/admin/api/settings")
async def admin_get_settings():
    """Get current runtime settings."""
    return runtime_settings


@app.post("/admin/api/settings")
async def admin_update_settings(body: SettingsUpdate):
    """Update runtime settings."""
    if body.timeout is not None:
        if body.timeout < 0:
            raise HTTPException(status_code=400, detail="Timeout must be >= 0 (0 = no limit)")
        runtime_settings["timeout"] = body.timeout
    return runtime_settings


# ============================================================================
# Static files
# ============================================================================

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/admin")
@app.get("/admin/")
async def admin_panel():
    """Serve the human operator dashboard."""
    admin_html = STATIC_DIR / "admin.html"
    if not admin_html.exists():
        return HTMLResponse("<h1>Admin panel not found</h1>", status_code=404)
    return FileResponse(admin_html)


@app.get("/")
async def root():
    """Root — basic server info."""
    return {
        "service": "AI API",
        "version": "1.1.0",
        "docs": "/docs",
        "admin": "/admin",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
        },
    }


# ============================================================================
# Helpers
# ============================================================================

async def _stream_response(
    request_id: str,
    created: int,
    model: str,
    content: Optional[str],
    tool_calls: Optional[List[Dict]],
):
    """Yields SSE chunks for streaming responses."""
    chunk_base = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }

    if tool_calls:
        # Stream tool call
        for i, tc in enumerate(tool_calls):
            fn = tc["function"]
            chunk = {
                **chunk_base,
                "choices": [{
                    "index": i,
                    "delta": {
                        "tool_calls": [{
                            "index": i,
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": fn["name"], "arguments": ""},
                        }]
                    },
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            # Stream arguments in pieces
            args = fn.get("arguments", "")
            for j in range(0, len(args), 8):
                piece = args[j:j+8]
                chunk = {
                    **chunk_base,
                    "choices": [{
                        "index": i,
                        "delta": {
                            "tool_calls": [{
                                "index": i,
                                "function": {"arguments": piece},
                            }]
                        },
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.02)
        # Finish
        chunk = {
            **chunk_base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    else:
        # Stream text content
        text = content or ""
        # First chunk: role
        yield f"data: {json.dumps({**chunk_base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
        # Stream content in pieces
        for i in range(0, len(text), 4):
            piece = text[i:i+4]
            yield f"data: {json.dumps({**chunk_base, 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]})}\n\n"
            await asyncio.sleep(0.02)
        # Final chunk: finish
        yield f"data: {json.dumps({**chunk_base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"

    yield "data: [DONE]\n\n"


async def _anthropic_stream(request_id: str, model: str, content: Optional[str], tool_calls: Optional[List[Dict]]):
    """Yields Anthropic-format SSE chunks."""
    msg_id = request_id.replace("chatcmpl-", "msg_").replace("msg_", "msg_")

    if tool_calls:
        # Tool use streaming
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function", {})
            args_str = fn.get("arguments", "{}")
            try:
                args_parsed = json.loads(args_str)
            except (json.JSONDecodeError, TypeError):
                args_parsed = {}

            # message_start
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': []}})}\n\n"
            # content_block_start
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': i, 'content_block': {'type': 'tool_use', 'id': tc.get('id', ''), 'name': fn.get('name', ''), 'input': {}}})}\n\n"
            # Stream arguments as deltas (in pieces)
            arg_items = list(args_parsed.items()) if args_parsed else []
            for key, val in arg_items:
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': i, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps({key: val}, ensure_ascii=False)}})}\n\n"
                await asyncio.sleep(0.02)
            # content_block_stop
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

        # message_delta + message_stop
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'tool_use'}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
    else:
        text = content or ""
        # message_start
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': []}})}\n\n"
        # content_block_start
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        # Stream text in pieces
        for i in range(0, len(text), 4):
            piece = text[i:i+4]
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': piece}})}\n\n"
            await asyncio.sleep(0.02)
        # content_block_stop
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        # message_delta + message_stop
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

def _archive_request(
    request_id: str,
    response: Optional[str] = None,
    tool_calls: Optional[List[Dict]] = None,
) -> None:
    """Move a request from pending → completed history."""
    if request_id not in pending_requests:
        return
    req = pending_requests.pop(request_id)
    req["completed_at"] = time.time()
    if response is not None:
        req["response"] = response
    if tool_calls is not None:
        req["tool_calls"] = tool_calls
    req["status"] = "completed" if (response or tool_calls) else req.get("status", "completed")
    req.pop("response_event", None)
    req.pop("response_data", None)
    completed_requests.append(req)
    while len(completed_requests) > MAX_HISTORY:
        completed_requests.pop(0)


def _estimate_tokens(messages: List[Message]) -> int:
    total = 0
    for m in messages:
        if m.content:
            total += len(m.content.split())
    return total


def _estimate_tokens_text(text: Optional[str]) -> int:
    return len(text.split()) if text else 0


def _estimate_tool_tokens(tool_calls: Optional[List[Dict]]) -> int:
    if not tool_calls:
        return 0
    total = 0
    for tc in tool_calls:
        fn = tc.get("function", {})
        total += len(fn.get("name", "").split())
        total += len(fn.get("arguments", "").split())
    return total


# ============================================================================
# Entry-point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI API Server")
    parser.add_argument("--host", default=HOST, help=f"Bind host (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Bind port (default: {PORT})")
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT,
                        help=f"Request timeout in seconds (default: {REQUEST_TIMEOUT})")
    args = parser.parse_args()

    runtime_settings["timeout"] = args.timeout

    display_host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    print(rf"""
╔══════════════════════════════════════════════════════════╗
║           AI API Server  v1.1                           ║
╠══════════════════════════════════════════════════════════╣
║  API Endpoint:  http://{display_host}:{args.port}/v1/chat/completions
║  Admin Panel:   http://{display_host}:{args.port}/admin
║  API Docs:      http://{display_host}:{args.port}/docs
║  Timeout:       {args.timeout}s
╠══════════════════════════════════════════════════════════╣
║  Supports: text responses + function/tool calls         ║
╚══════════════════════════════════════════════════════════╝
    """)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
