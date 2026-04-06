This project is an LLM Proxy that supports the following features:
1. Can proxy various LLM Models
2. Different protocols (e.g., openai and claude) are distinguished by base URL: OpenAI under /openai, Claude under /claude
3. A YAML config file maps client-facing model names (mapped) to real upstream model names (source), along with the upstream endpoint
4. Auth: if access_token is configured for a model, the proxy replaces the client token; otherwise, the client token is passed through
5. The real endpoint is unaware of the proxy — all traffic appears to come directly from the proxy
6. The system runs in Kubernetes — Dockerfile and Helm chart are provided
7. Cross-provider API translation: convert between OpenAI Chat, OpenAI Responses, Claude, and Gemini API formats

## Tech Stack

- Language: Go 1.22
- HTTP: standard `net/http` library
- Config parsing: `gopkg.in/yaml.v3`
- Container: Docker (multi-stage build, alpine base)
- Kubernetes: Helm chart in `deployments/helm/llm-proxy/`

## Project Structure

```
cmd/server/main.go                    # Entrypoint — HTTP server, routing, dialect registration
internal/config/                      # Config loading and model lookup
internal/proxy/handler.go             # HTTP handler, SSE streaming, translation routing
internal/proxy/rewriter.go            # Request rewriting (model name, auth headers)
internal/translate/                   # Translation layer
  canonical/types.go                  # Dialect-neutral request/response types
  dialect.go                          # Dialect + StreamDecoder/StreamEncoder interfaces
  registry.go                         # Global dialect registry (Register/Get)
  oaichat/dialect.go                  # OpenAI Chat Completions dialect
  oairesponses/dialect.go             # OpenAI Responses API dialect (Codex CLI 2025)
  claude/dialect.go                   # Anthropic Claude Messages API dialect
  gemini/dialect.go                   # Google Gemini GenerateContent dialect
config/config.yaml                    # Example configuration
deployments/helm/llm-proxy/           # Helm chart for Kubernetes
docs/                                 # Project documentation (see docs/index.md)
```

## Development

```bash
go mod tidy          # install dependencies
go test ./...        # run all tests
go build -o llm-proxy ./cmd/server
./llm-proxy --config config/config.yaml
```

## Config Fields

| Field | Description |
|-------|-------------|
| `mapped` | Client-facing model name |
| `source` | Actual model name sent to the upstream endpoint |
| `mapped_base_url` | Proxy path prefix (e.g., `openai` → routes `/openai/*`). Any string; no hardcoded values. |
| `original_base_url` | Real upstream base URL (e.g., `https://api.openai.com`) |
| `access_token` | Optional. If set, replaces client token. If empty, client token is passed through. |
| `auth_header` | Header name for the auth token (e.g., `Authorization`, `x-api-key`, `x-goog-api-key`) |
| `auth_scheme` | Token prefix (e.g., `Bearer`). Empty means raw token with no prefix. |
| `incoming_dialect` | Optional. API dialect from the client (e.g., `openai-chat`, `claude`, `openai-responses`, `gemini`). |
| `upstream_dialect` | Optional. API dialect for the upstream service. When both dialects are set, translation is enabled. |

## Key Design Notes

- `mapped_base_url` and `original_base_url` are fully config-driven. Adding a new provider requires only a new config entry — no code changes.
- Routes are registered dynamically at startup based on unique `mapped_base_url` values in config.
- Request body is fully buffered to allow JSON `model` field rewriting before forwarding.
- Responses are streamed chunk-by-chunk using `http.Flusher` for SSE support.
- When `access_token` is configured: all client auth headers are stripped, and the configured auth header is set.
- When no `access_token`: all client headers (including auth) are passed through unchanged.
- `/healthz` endpoint returns 200 OK for Kubernetes liveness/readiness probes.

## Translation Layer

When `incoming_dialect` and `upstream_dialect` are both set on a model config, the proxy performs full API translation:

1. Incoming request is decoded into a dialect-neutral canonical format (`internal/translate/canonical`).
2. The canonical request is re-encoded into the upstream dialect's format.
3. The upstream response is decoded back to canonical, then re-encoded into the incoming dialect's format.
4. Streaming responses use `StreamDecoder` (upstream) and `StreamEncoder` (client) interfaces for event-by-event translation.

### Supported Dialects

| Dialect ID | API | Used by |
|------------|-----|---------|
| `openai-chat` | Chat Completions `/v1/chat/completions` | Most OpenAI-compatible clients |
| `openai-responses` | Responses API `/v1/responses` | Codex CLI 2025 |
| `claude` | Messages API `/v1/messages` | Claude Code, Anthropic SDK |
| `gemini` | GenerateContent `/v1beta/models/{model}:generateContent` | Gemini SDK |

### Translation Examples

```yaml
# Use Claude from any OpenAI Chat client:
- mapped: claude-via-openai
  source: claude-3-5-sonnet-20241022
  mapped_base_url: openai-compat
  original_base_url: https://api.anthropic.com
  access_token: "sk-ant-..."
  auth_header: x-api-key
  auth_scheme: ""
  incoming_dialect: openai-chat
  upstream_dialect: claude

# Use GPT-4o from Claude Code:
- mapped: gpt4-via-claude
  source: gpt-4o
  mapped_base_url: claude-compat
  original_base_url: https://api.openai.com
  access_token: "sk-..."
  auth_header: Authorization
  auth_scheme: Bearer
  incoming_dialect: claude
  upstream_dialect: openai-chat
```
