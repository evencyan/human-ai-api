# 能工智人API 项目文档

## 项目概述

一个 OpenAI/Anthropic 双兼容的 API 服务端。**核心机制**：API 请求到达后不立即返回，而是进入等待队列；人工操作员在 Web 管理面板中手动撰写回复，回复后 API 才返回结果。调用方全程感知不到背后是人工操作。

## 架构

```
┌─────────┐     POST /v1/chat/completions     ┌──────────────┐
│ 调用方   │ ──────────────────────────────→  │  能工智人API   │
│(Claude   │                                   │  (FastAPI)   │
│ Code/    │ ←──── OpenAI/Anthropic JSON ────  │              │
│ curl)    │      (阻塞直到人工回复)             │  等待队列     │
└─────────┘                                   │  ↓           │
                                              │ 管理面板 /admin│
                                              │  ↓           │
                                              │ 人工输入回复   │
                                              └──────────────┘
```

- **请求阻塞**：使用 `asyncio.Event` 挂起请求，人工回复后 `event.set()` 唤醒
- **超时控制**：运行时可变，0 表示无限制
- **存储**：纯内存（`dict` + `list`），重启丢失所有数据

## 文件结构

```
human-ai-api/
├── server.py          # 主服务端，所有逻辑都在这里
├── static/admin.html  # 单文件 Web 管理面板
├── requirements.txt   # fastapi, uvicorn, pydantic
├── README.md          # 面向用户的说明
├── CLAUDE.md          # 本文件，面向 AI 的完整项目文档
└── .gitignore
```

## API 端点

### 外部接口（调用方使用）

| 方法 | 路径 | 格式 | 说明 |
|------|------|------|------|
| GET | `/v1/models` | OpenAI | 返回模型列表 |
| POST | `/v1/chat/completions` | OpenAI | 聊天补全，阻塞等待 |
| POST | `/v1/messages` | Anthropic | 消息 API，内部转换为 OpenAI 格式处理 |

### 管理接口（管理面板使用）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/admin` | 管理面板首页 |
| GET | `/admin/api/requests` | 获取待处理/已完成请求 |
| POST | `/admin/api/respond` | 提交回复（文本或工具调用） |
| POST | `/admin/api/skip` | 跳过请求（调用方收到 500 错误） |
| GET | `/admin/api/stats` | 服务器统计 |
| GET/POST | `/admin/api/settings` | 运行时设置读写 |
| GET | `/docs` | FastAPI 自动文档 |

## 模型

当前返回三个模型（全部由人工操作，无实质区别）：

| ID | owned_by | context_window |
|----|----------|---------------|
| gpt-4 | openai | 131072 |
| gpt-4o | openai | 131072 |
| claude-4 | anthropic | 200000 |

## 关键设计决策

### 1. OpenAI ↔ Anthropic 格式转换

Claude Code 使用 Anthropic Messages API 格式。`_anthropic_to_internal()` 函数负责转换请求：
- Anthropic 的 `content` 可以是字符串或内容块数组，需逐类型处理（text/tool_use/tool_result）
- `tool_choice: "any"` → `"required"`，`"auto"` 保持，指定工具名 → `{"type":"function","function":{"name":"..."}}`
- 系统提示 `system` 字段转为 `role: system` 消息

`_internal_to_anthropic()` 负责转换响应：
- 文本回复 → `content: [{type: "text", text: "..."}]`, `stop_reason: "end_turn"`
- 工具调用 → `content: [{type: "tool_use", ...}]`, `stop_reason: "tool_use"`

### 2. 流式输出（SSE）

两种格式的流式均支持：

**OpenAI SSE**：`data: {JSON}\n\n` 格式，最后 `data: [DONE]\n\n`
- 首块含 `delta: {role: "assistant"}`
- 内容分 4 字符一块发送
- 末块 `finish_reason: "stop"` / `"tool_calls"`

**Anthropic SSE**：`event: <type>\ndata: {JSON}\n\n` 格式
- message_start → content_block_start → content_block_delta (循环) → content_block_stop → message_delta → message_stop
- 工具调用的 delta type 为 `input_json_delta`

### 3. 工具调用（Tool Calling）

- `tool_choice=required` 或指定工具 → 管理面板锁定工具调用模式，文本按钮灰色不可点击
- `tool_choice=any` / `auto` → 操作员可选文本或工具调用
- 服务端双重检查：`tc_is_required` 时拒绝纯文本回复，返回 400
- 工具调用响应中 `content: null`
- Anthropic 工具格式：`{name, description, input_schema}` ↔ OpenAI：`{type:"function", function:{name, description, parameters}}`

### 4. 去"人类"痕迹

外部 API 所有 "human" 字样已清除：
- 模型 ID 从 `human-*` 改为标准名称
- `owned_by` 为 `openai`/`anthropic`
- 错误消息不含 "human operator" 等字样
- 服务名显示为 "能工智人API"
- **管理面板内部**仍可保留操作相关提示

### 5. 运行时设置

通过 `/admin/api/settings` API 和管理面板设置弹窗修改：
- `timeout`：超时秒数，0 = 无限
- 客户端设置：自动加载下一个待回复、通知声音（存储在 localStorage）

### 6. 管理面板设计

- **配色**：刻意避免 AI 风格（无蓝色渐变、无 Inter 字体、无 emoji 装饰），采用中性灰色调
- **深浅色**：CSS `prefers-color-scheme` 自动跟随系统
- **中英文**：`I18N` 对象管理，`localStorage` 持久化语言偏好
- **快捷键**：`Enter` 发送，`Shift+Enter` / `Ctrl+Enter` 换行
- **回复框常驻**：不会因无请求而关闭，无请求时按钮变灰
- **工具折叠**：大量工具时默认收起，点击展开

## 已知问题与处理

- **`stream=true` 不返回流式** → 已修复，实现完整 SSE
- **33 个工具显示为 ?** → Anthropic 工具格式与 OpenAI 不同，已适配两种格式读取 name/description
- **`tool_choice: any` 被误判为 required** → 已修复，只有 `required` 或指定工具才强制
- **base_url 双重 `/v1`** → 提示用户软件会在 base_url 后自动拼接 `/v1`
- **跳过请求后错误信息暴露人工操作** → 改为 "Request could not be processed"

## 开发约定

- 不默认启动服务端，由用户手动运行
- 不自动运行测试
- 对外 API 不出现 "human" 字样
- 管理面板用 `i18n` 对象统一管理文字
- `server.py` 为单文件，所有逻辑在一处
