package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/iopanda/llm-proxy/internal/config"
)

// parseModelName extracts the model field from a JSON request body.
func parseModelName(body []byte) (string, error) {
	var payload struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", fmt.Errorf("parse request body: %w", err)
	}
	if payload.Model == "" {
		return "", fmt.Errorf("model field is required")
	}
	return payload.Model, nil
}

// buildUpstreamRequest constructs the forwarded request to the upstream LLM endpoint.
// It rewrites the model name, manages auth headers, and preserves all other request attributes.
func buildUpstreamRequest(r *http.Request, body []byte, modelCfg *config.ModelConfig, mappedBaseURL string) (*http.Request, error) {
	newBody, err := rewriteModelName(body, modelCfg.BackendModelName)
	if err != nil {
		return nil, err
	}

	upstreamPath := strings.TrimPrefix(r.URL.Path, mappedBaseURL)
	upstreamURL := strings.TrimRight(modelCfg.BackendBaseURL, "/") + upstreamPath
	if r.URL.RawQuery != "" {
		upstreamURL += "?" + r.URL.RawQuery
	}

	upstreamReq, err := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL, bytes.NewReader(newBody))
	if err != nil {
		return nil, fmt.Errorf("create upstream request: %w", err)
	}

	if modelCfg.BackendAPIKey != "" {
		// Configured token: strip all client auth headers, then set the configured one.
		skipHeaders := authHeadersToSkip(modelCfg.BackendAuthHeader)
		copyRequestHeaders(upstreamReq.Header, r.Header, skipHeaders)
		setAuthHeader(upstreamReq, modelCfg.BackendAuthHeader, modelCfg.BackendAuthSchema, modelCfg.BackendAPIKey)
	} else {
		// Passthrough: copy all headers including auth, unchanged.
		copyRequestHeaders(upstreamReq.Header, r.Header, nil)
	}

	upstreamReq.ContentLength = int64(len(newBody))
	upstreamReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(newBody)))

	return upstreamReq, nil
}

// rewriteModelName replaces the model field value in a JSON body.
func rewriteModelName(body []byte, sourceName string) ([]byte, error) {
	if len(body) == 0 {
		return body, nil
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("parse request body: %w", err)
	}
	payload["model"] = sourceName
	return json.Marshal(payload)
}

// setAuthHeader applies the auth token to the upstream request using the configured header and scheme.
func setAuthHeader(r *http.Request, authHeader, authScheme, token string) {
	if authHeader == "" || token == "" {
		return
	}
	if authScheme != "" {
		r.Header.Set(authHeader, authScheme+" "+token)
	} else {
		r.Header.Set(authHeader, token)
	}
}

// authHeadersToSkip returns the set of header names to strip when a configured token is used.
// Always includes the well-known auth headers plus the model's own configured auth header.
func authHeadersToSkip(configuredAuthHeader string) map[string]bool {
	skip := map[string]bool{
		"Authorization": true,
		"X-Api-Key":     true,
	}
	if configuredAuthHeader != "" {
		skip[http.CanonicalHeaderKey(configuredAuthHeader)] = true
	}
	return skip
}

// buildTranslatedRequest constructs the upstream request with a pre-translated body.
// It handles auth header management but skips the model-name rewriting (already done by the dialect).
func buildTranslatedRequest(r *http.Request, body []byte, modelCfg *config.ModelConfig, upstreamURL string) (*http.Request, error) {
	upstreamReq, err := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create translated upstream request: %w", err)
	}

	if modelCfg.BackendAPIKey != "" {
		skipHeaders := authHeadersToSkip(modelCfg.BackendAuthHeader)
		copyRequestHeaders(upstreamReq.Header, r.Header, skipHeaders)
		setAuthHeader(upstreamReq, modelCfg.BackendAuthHeader, modelCfg.BackendAuthSchema, modelCfg.BackendAPIKey)
	} else {
		copyRequestHeaders(upstreamReq.Header, r.Header, nil)
	}

	upstreamReq.ContentLength = int64(len(body))
	upstreamReq.Header.Set("Content-Length", fmt.Sprintf("%d", len(body)))
	upstreamReq.Header.Set("Content-Type", "application/json")

	return upstreamReq, nil
}

// copyRequestHeaders copies headers from src to dst, skipping hop-by-hop headers and
// any headers in the skipHeaders set. Pass nil skipHeaders to copy all non-hop-by-hop headers.
func copyRequestHeaders(dst, src http.Header, skipHeaders map[string]bool) {
	hopByHop := map[string]bool{
		"Content-Length":    true,
		"Host":              true,
		"Connection":        true,
		"Transfer-Encoding": true,
		"Accept-Encoding":   true, // let Go http.Client manage compression/decompression
	}
	for key, values := range src {
		canonical := http.CanonicalHeaderKey(key)
		if hopByHop[canonical] {
			continue
		}
		if skipHeaders[canonical] {
			continue
		}
		for _, v := range values {
			dst.Add(key, v)
		}
	}
}
