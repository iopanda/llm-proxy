# LLM Proxy

A lightweight reverse proxy for LLM APIs. It routes requests to different upstream providers, rewrites model names and auth tokens, and translates between API formats — all driven by a single YAML config file.

## Features

- **Proxy mode**: forward requests to any LLM API, rewriting the model name and auth token transparently.
- **Translation mode**: convert between API formats on the fly (e.g., use Claude from an OpenAI-compatible client, or use GPT-4o from Claude Code).
- **Streaming support**: SSE responses are streamed chunk-by-chunk with zero buffering.
- **Config-driven routing**: add a new provider or model alias by editing the config file — no code changes needed.
- **Kubernetes-ready**: includes a Dockerfile and Helm chart.

## Supported API Dialects

| Dialect ID | API | Used by |
|---|---|---|
| `openai-chat` | Chat Completions `/v1/chat/completions` | OpenAI SDK, most OpenAI-compatible clients |
| `openai-responses` | Responses API `/v1/responses` | Codex CLI (2025) |
| `claude` | Messages API `/v1/messages` | Claude Code, Anthropic SDK |
| `bedrock` | AWS Bedrock InvokeModel | AWS SDK, Bedrock clients |

## Quick Start

**1. Install**

```bash
git clone <repo>
cd llm-proxy
go build -o llm-proxy ./cmd/server
```

**2. Configure**

Copy the example config and fill in your credentials:

```bash
cp config/config.example.yaml config/config.yaml
```

**3. Run**

```bash
./llm-proxy --config config/config.yaml
```

The server starts on port `8080` by default. A `/healthz` endpoint returns `200 OK` for health checks.

## Configuration

```yaml
server:
  port: 8080

models:
  - mapped: <client-facing model name>
    original_model_name: <real upstream model name>
    mapped_base_url: <route prefix>       # requests go to /<mapped_base_url>/*
    original_base_url: <upstream base URL>
    access_token: ""                      # leave empty to pass client token through
    auth_header: Authorization            # header used to send the token upstream
    auth_scheme: Bearer                   # token prefix; empty = raw token
    incoming_dialect: ""                  # set for translation (see below)
    upstream_dialect: ""                  # set for translation (see below)
```

### Proxy Mode (no translation)

Forward requests as-is, only rewriting the model name and auth token:

```yaml
models:
  - mapped: gpt-4-proxy
    original_model_name: gpt-4
    mapped_base_url: openai
    original_base_url: https://api.openai.com
    access_token: "sk-..."
    auth_header: Authorization
    auth_scheme: Bearer
```

Client points at `http://localhost:8080/openai/v1/chat/completions` with model `gpt-4-proxy`. The proxy forwards to `https://api.openai.com/v1/chat/completions` with model `gpt-4`.

### Translation Mode

Set both `incoming_dialect` and `upstream_dialect` to enable API format translation:

```yaml
models:
  # Use Claude from any OpenAI Chat client:
  - mapped: claude-via-openai
    original_model_name: claude-3-5-sonnet-20241022
    mapped_base_url: openai-compat
    original_base_url: https://api.anthropic.com
    access_token: "sk-ant-..."
    auth_header: x-api-key
    auth_scheme: ""
    incoming_dialect: openai-chat
    upstream_dialect: claude

  # Use GPT-4o from Claude Code:
  - mapped: gpt4-via-claude
    original_model_name: gpt-4o
    mapped_base_url: claude-compat
    original_base_url: https://api.openai.com
    access_token: "sk-..."
    auth_header: Authorization
    auth_scheme: Bearer
    incoming_dialect: claude
    upstream_dialect: openai-chat

```

### Config Fields

| Field | Required | Description |
|---|---|---|
| `mapped` | Yes | Model name the client sends |
| `original_model_name` | Yes | Actual model name sent to upstream |
| `mapped_base_url` | Yes | URL path prefix for this group (e.g., `openai` → routes `/openai/*`) |
| `original_base_url` | Yes | Real upstream base URL |
| `access_token` | No | If set, replaces the client's token; if empty, client token is forwarded |
| `auth_header` | No | Header name for the auth token |
| `auth_scheme` | No | Token prefix (e.g., `Bearer`). Empty = raw token |
| `incoming_dialect` | No | API format from the client. Required for translation. |
| `upstream_dialect` | No | API format for the upstream. Required for translation. |

## How Translation Works

When both dialects are configured, the proxy:

1. Decodes the incoming request into a dialect-neutral canonical format.
2. Re-encodes it into the upstream dialect's format.
3. Sends it to the upstream and decodes the response back to canonical.
4. Re-encodes the response into the incoming dialect's format for the client.

For streaming, this happens event-by-event with no full-response buffering.

## Routing

Routes are registered dynamically at startup based on the unique `mapped_base_url` values in the config. The URL structure is:

```
http://localhost:8080/<mapped_base_url>/<api-path>
```

For example, with `mapped_base_url: openai`, configure your client's base URL as `http://localhost:8080/openai`.

## Auth

| Scenario | Behavior |
|---|---|
| `access_token` is set | All client auth headers are stripped; the configured token is sent upstream |
| `access_token` is empty | All client headers (including auth) are forwarded unchanged |

## Development

```bash
go mod tidy        # install dependencies
go test ./...      # run all tests
go build -o llm-proxy ./cmd/server
./llm-proxy --config config/config.yaml
```

See [`docs/`](docs/index.md) for architecture, development, and deployment documentation.

## Deployment

A Dockerfile and Helm chart are provided under `deployments/helm/llm-proxy/`. See [`docs/deploy/deployment.md`](docs/deploy/deployment.md) for details.
