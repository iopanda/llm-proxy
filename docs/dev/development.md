# Development Guide

## Prerequisites

- Go 1.22+
- Docker (for container builds)
- `helm` CLI (for chart validation)

## Project Structure

```
llm-proxy/
├── cmd/server/main.go              # Application entrypoint
├── internal/
│   ├── config/
│   │   ├── config.go               # Config loading and model lookup
│   │   └── config_test.go
│   └── proxy/
│       ├── handler.go              # HTTP handler and streaming
│       ├── handler_test.go
│       ├── rewriter.go             # Request rewriting (model, auth, headers)
│       └── rewriter_test.go
├── config/config.yaml              # Example configuration
├── deployments/helm/llm-proxy/     # Helm chart
├── docs/                           # Project documentation
├── Dockerfile
├── go.mod
└── go.sum
```

## Getting Started

```bash
# Install dependencies
go mod tidy

# Run tests
go test ./...

# Build binary
go build -o llm-proxy ./cmd/server

# Run locally
./llm-proxy --config config/config.yaml
```

## Running Tests

```bash
# All tests
go test ./...

# With verbose output
go test -v ./...

# Specific package
go test ./internal/proxy/...
go test ./internal/config/...
```

## Configuration

Edit `config/config.yaml` to add model mappings. See [Operations Guide](../operate/operations.md) for the full config reference.

## Adding a New Protocol

To add support for a new LLM protocol (e.g., `gemini`):

1. Add the protocol constant in `internal/proxy/rewriter.go`.
2. Implement protocol-specific auth header logic in `setAuthHeader`.
3. Add the route in `cmd/server/main.go`.
4. Add a new `Handler` method (e.g., `Gemini()`).

## Code Conventions

- Follow standard Go conventions (`gofmt`, `go vet`).
- All tests use `testing` standard library — no external test frameworks.
- Internal functions (lowercase) are tested in `package proxy`.
- Public API is tested in `package proxy_test` for black-box coverage.
