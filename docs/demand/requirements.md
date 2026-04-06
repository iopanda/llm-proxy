# Requirements

## Functional Requirements

### FR-1: Multi-Protocol Proxying

The system must support proxying requests to LLM endpoints using different protocols:

- **OpenAI-compatible**: requests handled under `/openai/*`
- **Anthropic Claude**: requests handled under `/claude/*`

Clients direct traffic to the appropriate base path based on the protocol of the target model.

### FR-2: Model Name Mapping

The system must maintain a mapping from **mapped names** (client-facing) to **source names** (upstream).

- Clients send requests using the mapped model name.
- The proxy rewrites the `model` field in the request body to the source name before forwarding.
- All other request fields are preserved unchanged.

Example: client sends `"model": "gpt-4-proxy"` → proxy forwards `"model": "gpt-4"`.

### FR-3: Auth Token Management

The system must manage authentication tokens per model:

- If a model has `access_token` configured, the proxy **replaces** the client's token with the configured token.
- If no `access_token` is configured, the proxy **passes through** the client's token unchanged.

Auth header conventions by protocol:

| Protocol | Header |
|----------|--------|
| openai   | `Authorization: Bearer <token>` |
| claude   | `x-api-key: <token>` |

### FR-4: Transparent Upstream

The upstream endpoint must be unaware of the proxy's existence:

- The proxy does not add proxy-identifying headers (e.g., `X-Forwarded-For`).
- All original request headers (except auth and hop-by-hop) are forwarded.
- The upstream sees requests as if they came directly from a client.

### FR-5: SSE Streaming Support

The system must support streaming responses (Server-Sent Events):

- Response chunks are flushed to the client as they arrive from upstream.
- `Content-Type: text/event-stream` responses are forwarded without buffering.

### FR-6: YAML Configuration

The system must be configurable via a YAML file at startup:

- Server port
- Model mappings (mapped name, source name, protocol, endpoint, optional access token)

### FR-7: Health Check Endpoint

The system must expose a `/healthz` HTTP endpoint that returns `200 OK` for liveness and readiness probes.

## Non-Functional Requirements

### NFR-1: Performance

- Must not introduce significant latency beyond network round-trip.
- Streaming responses must be forwarded with low buffering overhead.

### NFR-2: Kubernetes-Ready

- Must be deployable as a Docker container.
- Must include a Helm chart for Kubernetes deployment.
- Must expose health check endpoints for liveness/readiness probes.

### NFR-3: Security

- Proxy itself does not require authentication (auth is managed per upstream model).
- Sensitive tokens in config should be managed via Kubernetes Secrets in production.
