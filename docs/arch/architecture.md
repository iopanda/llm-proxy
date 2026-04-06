# Architecture

## System Overview

```
Client
  │
  │  POST /openai/v1/chat/completions
  │  Authorization: Bearer <client-token>
  │  {"model": "gpt-4-proxy", ...}
  │
  ▼
┌─────────────────────────────────┐
│           LLM Proxy             │
│                                 │
│  ┌─────────┐   ┌─────────────┐  │
│  │ Router  │──▶│   Handler   │  │
│  └─────────┘   └──────┬──────┘  │
│                        │        │
│                ┌───────▼──────┐ │
│                │   Rewriter   │ │
│                │ - model name │ │
│                │ - auth token │ │
│                └───────┬──────┘ │
└────────────────────────┼────────┘
                         │
                         │  POST /v1/chat/completions
                         │  Authorization: Bearer <config-token>
                         │  {"model": "gpt-4", ...}
                         ▼
                  Upstream LLM API
                 (OpenAI / Claude)
```

## Components

### Router

`cmd/server/main.go` sets up the HTTP server and routes:

| Path prefix | Handler |
|-------------|---------|
| `/openai/*` | OpenAI protocol handler |
| `/claude/*` | Claude protocol handler |
| `/healthz`  | Health check |

### Handler (`internal/proxy/handler.go`)

Orchestrates the proxy flow for each request:

1. Read and buffer the request body.
2. Extract the `model` field from the JSON body.
3. Look up the model config by protocol + mapped name.
4. Delegate to Rewriter to build the upstream request.
5. Forward to upstream using a standard `http.Client`.
6. Stream the response back to the client chunk-by-chunk.

### Rewriter (`internal/proxy/rewriter.go`)

Handles all request transformation logic:

- **`parseModelName`**: extracts the model name from the request JSON body.
- **`buildUpstreamRequest`**: constructs the upstream `*http.Request` with:
  - Rewritten URL (protocol prefix stripped, upstream base URL prepended).
  - Rewritten `model` field in the body.
  - Auth header set per protocol and config.
  - All safe request headers copied.
- **`extractClientToken`**: reads the token from the client request (supports both `Authorization: Bearer` and `x-api-key`).
- **`setAuthHeader`**: applies token to the upstream request in the correct header for the protocol.

### Config (`internal/config/config.go`)

Loads and parses the YAML config file. Provides `FindModel(protocol, mappedName)` for O(n) model lookup.

## Data Flow

### Normal (non-streaming) Request

```
Client → Proxy: POST /openai/v1/chat/completions, body={"model":"gpt-4-proxy",...}
Proxy: read body, parse model="gpt-4-proxy"
Proxy: lookup config → source="gpt-4", endpoint="https://api.openai.com", token="sk-..."
Proxy: rewrite body → {"model":"gpt-4",...}, set Authorization: Bearer sk-...
Proxy → Upstream: POST https://api.openai.com/v1/chat/completions
Upstream → Proxy: 200 OK, {"choices":[...]}
Proxy → Client: 200 OK, {"choices":[...]}
```

### Streaming (SSE) Request

```
Client → Proxy: POST /openai/v1/chat/completions, body={"model":"gpt-4-proxy","stream":true,...}
Proxy → Upstream: (rewritten request)
Upstream → Proxy: 200 OK, Content-Type: text/event-stream
  data: {"choices":[{"delta":{"content":"Hello"}}]}
  data: [DONE]
Proxy → Client: (each chunk flushed immediately via http.Flusher)
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Standard `net/http` (no framework) | Minimal dependencies, sufficient for this use case |
| Buffer full request body | Required to rewrite JSON `model` field before forwarding |
| Per-chunk flushing | Ensures low latency for SSE streaming |
| Hop-by-hop header filtering | Prevents proxy-specific headers from leaking to upstream |
| Config token overrides client token | Allows centralized credential management without client changes |
