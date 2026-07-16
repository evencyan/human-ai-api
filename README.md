# Human AI API

An OpenAI/Anthropic-compatible API server that lets a human operator respond to chat completion requests through a web dashboard.

The caller sees a normal AI API — the operator sees an admin panel.

## Features

- **OpenAI-compatible** — `/v1/chat/completions`, `/v1/models`
- **Anthropic-compatible** — `/v1/messages` (works with Claude Code)
- **Tool calling** — function/tool call requests, including `tool_choice`
- **Streaming** — SSE streaming for both OpenAI and Anthropic formats
- **Web admin panel** — dark/light theme, i18n (zh/en), reply via text or tool call
- **Settings** — configurable timeout, auto-load next request, notification sound

## Quick Start

```bash
pip install -r requirements.txt
python server.py --port 8001
```

Open `http://localhost:8001/admin` for the operator panel.

## API Usage

**Base URL:** `http://localhost:8001/v1`

**API Key:** any string (no authentication required)

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello!"}]}'
```

## Claude Code Setup

Set these environment variables:

```
ANTHROPIC_BASE_URL=http://localhost:8001
ANTHROPIC_MODEL=claude-4
ANTHROPIC_AUTH_TOKEN=sk-any
```
