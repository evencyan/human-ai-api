# 能工智人API

一个 OpenAI/Anthropic 兼容的 API 服务端，背后由人工操作员通过网页管理面板回复消息。调用方以为在调用真正的 AI API，实际上每条回复都是人在写。

> **注意：本项目代码完全由 AI（Claude Code）生成，未经人工编写或审查。**

## 功能

- **OpenAI 兼容** — `/v1/chat/completions`、`/v1/models`
- **Anthropic 兼容** — `/v1/messages`（可用作 Claude Code 后端）
- **工具调用** — 支持 function call / tool call，包括 `tool_choice`
- **流式输出** — 同时支持 OpenAI 和 Anthropic 格式的 SSE 流式返回
- **网页管理面板** — 深浅色主题、中英文切换、文本/工具调用双模式回复
- **设置面板** — 可调超时时间、自动加载下一个请求、新消息提示音

## 快速开始

```bash
pip install -r requirements.txt
python server.py --port 8001
```

打开 `http://localhost:8001/admin` 进入管理面板。

## 接口调用

**Base URL:** `http://localhost:8001/v1`

**API Key:** 任意字符串（无需鉴权）

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"你好"}]}'
```

## 用作 Claude Code 后端

修改 `~/.claude/settings.json` 中的环境变量：

```
ANTHROPIC_BASE_URL=http://localhost:8001
ANTHROPIC_MODEL=claude-4
ANTHROPIC_AUTH_TOKEN=sk-任意
```

## 工作流程

```
调用方 ──API 请求──→ 服务端（阻塞等待）
                       ↓
操作员 ──查看管理面板──→ 输入回复 / 工具调用
                       ↓
调用方 ←──JSON 响应── 服务端返回结果
```

## 模型

| 模型 ID | 上下文 |
|---------|--------|
| `gpt-4` | 128K |
| `gpt-4o` | 128K |
| `claude-4` | 200K |
