# Operations Guide

## Configuration Reference

The proxy is configured via a YAML file (default: `config/config.yaml`).

```yaml
server:
  port: 8080          # HTTP listening port (default: 8080)

models:
  - mapped: gpt-4-proxy             # Name clients use in the "model" field
    source: gpt-4                   # Actual model name sent to upstream
    protocol: openai                # "openai" or "claude"
    endpoint: https://api.openai.com  # Upstream base URL (no trailing slash)
    access_token: ""                # If empty, client token is passed through
```

### Protocol Values

| Value | Auth Header | Use For |
|-------|-------------|---------|
| `openai` | `Authorization: Bearer <token>` | OpenAI and OpenAI-compatible APIs |
| `claude` | `x-api-key: <token>` | Anthropic Claude API |

### Auth Token Behavior

| `access_token` in config | Behavior |
|--------------------------|----------|
| Set (non-empty) | Proxy replaces client token with configured token |
| Not set (empty) | Proxy passes client token through unchanged |

## Client Usage

### OpenAI-compatible Endpoint

```bash
curl http://localhost:8080/openai/v1/chat/completions \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4-proxy", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Claude Endpoint

```bash
curl http://localhost:8080/claude/v1/messages \
  -H "x-api-key: <your-token>" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-proxy", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

### Streaming (SSE)

Add `"stream": true` to the request body. The proxy forwards chunks as they arrive from upstream.

```bash
curl http://localhost:8080/openai/v1/chat/completions \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -N \
  -d '{"model": "gpt-4-proxy", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

## Error Reference

| HTTP Status | Cause |
|-------------|-------|
| 400 Bad Request | Request body is not valid JSON, or missing `model` field |
| 404 Not Found | `model` value in the request is not in the proxy config |
| 502 Bad Gateway | Upstream endpoint is unreachable or returned a connection error |
| Upstream status | All other upstream status codes are forwarded as-is |

## Logging

The proxy logs to stdout using Go's standard `log` package. Log lines are prefixed with timestamp and level:

```
2024/01/15 10:00:00 llm-proxy listening on :8080
2024/01/15 10:00:05 ERROR upstream request: dial tcp: connection refused
```

## Troubleshooting

**404 on valid model name**: verify the `protocol` in the config matches the path prefix used (`/openai/` vs `/claude/`).

**Upstream receives wrong token**: check if `access_token` is accidentally set in config. Remove or empty it to enable passthrough.

**Streaming not working**: ensure the client does not set `Accept-Encoding: gzip` (which would cause buffering). The proxy forwards `Content-Type: text/event-stream` from upstream.
