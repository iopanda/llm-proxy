package proxy_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/iopanda/llm-proxy/internal/config"
	"github.com/iopanda/llm-proxy/internal/proxy"

	// Register translation dialects for integration tests.
	_ "github.com/iopanda/llm-proxy/internal/translate/claude"
	_ "github.com/iopanda/llm-proxy/internal/translate/oaichat"
)

type capturedRequest struct {
	model   string
	headers http.Header
}

func newUpstreamServer(t *testing.T, response string, statusCode int) (*httptest.Server, *capturedRequest) {
	t.Helper()
	captured := &capturedRequest{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured.headers = r.Header.Clone()

		var payload map[string]interface{}
		if err := json.Unmarshal(body, &payload); err == nil {
			if m, ok := payload["model"].(string); ok {
				captured.model = m
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		w.Write([]byte(response))
	}))
	t.Cleanup(srv.Close)
	return srv, captured
}

func newProxyServer(t *testing.T, cfg *config.Config) *httptest.Server {
	t.Helper()
	h := proxy.NewHandler(cfg)
	mux := http.NewServeMux()
	for _, baseURL := range cfg.UniqueFrontendBaseURLs() {
		mux.Handle(baseURL+"/", h.For(baseURL))
	}
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestHandler_OpenAI_ConfigToken_ReplacesClientToken(t *testing.T) {
	upstream, captured := newUpstreamServer(t, `{"choices":[{"message":{"content":"hi"}}]}`, 200)

	cfg := &config.Config{
		Models: []config.ModelConfig{
			{
				FrontendModelName: "my-gpt4", BackendModelName: "gpt-4", FrontendBaseURL: "/openai",
				BackendBaseURL: upstream.URL, BackendAPIKey: "sk-config",
				BackendAuthHeader: "Authorization", BackendAuthSchema: "Bearer",
			},
		},
	}
	proxyServer := newProxyServer(t, cfg)

	reqBody := `{"model":"my-gpt4","messages":[{"role":"user","content":"hello"}]}`
	req, _ := http.NewRequest("POST", proxyServer.URL+"/openai/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-client")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status=%d, want 200", resp.StatusCode)
	}
	if captured.model != "gpt-4" {
		t.Errorf("upstream model=%q, want gpt-4", captured.model)
	}
	if got := captured.headers.Get("Authorization"); got != "Bearer sk-config" {
		t.Errorf("upstream Authorization=%q, want Bearer sk-config (config token must override client)", got)
	}
}

func TestHandler_OpenAI_NoConfigToken_PassthroughClientToken(t *testing.T) {
	upstream, captured := newUpstreamServer(t, `{"choices":[]}`, 200)

	cfg := &config.Config{
		Models: []config.ModelConfig{
			{
				FrontendModelName: "my-gpt4", BackendModelName: "gpt-4", FrontendBaseURL: "/openai",
				BackendBaseURL: upstream.URL,
				BackendAuthHeader: "Authorization", BackendAuthSchema: "Bearer",
			},
		},
	}
	proxyServer := newProxyServer(t, cfg)

	reqBody := `{"model":"my-gpt4","messages":[]}`
	req, _ := http.NewRequest("POST", proxyServer.URL+"/openai/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer sk-client")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if got := captured.headers.Get("Authorization"); got != "Bearer sk-client" {
		t.Errorf("upstream Authorization=%q, want Bearer sk-client (should pass through)", got)
	}
}

func TestHandler_Claude_XApiKey(t *testing.T) {
	upstream, captured := newUpstreamServer(t, `{"content":[{"text":"hi"}]}`, 200)

	cfg := &config.Config{
		Models: []config.ModelConfig{
			{
				FrontendModelName: "my-claude", BackendModelName: "claude-3-5-sonnet-20241022", FrontendBaseURL: "/claude",
				BackendBaseURL: upstream.URL, BackendAPIKey: "sk-ant-config",
				BackendAuthHeader: "x-api-key", BackendAuthSchema: "",
			},
		},
	}
	proxyServer := newProxyServer(t, cfg)

	reqBody := `{"model":"my-claude","messages":[{"role":"user","content":"hello"}],"max_tokens":100}`
	req, _ := http.NewRequest("POST", proxyServer.URL+"/claude/v1/messages", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "sk-ant-client")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status=%d, want 200", resp.StatusCode)
	}
	if captured.model != "claude-3-5-sonnet-20241022" {
		t.Errorf("upstream model=%q, want claude-3-5-sonnet-20241022", captured.model)
	}
	if got := captured.headers.Get("x-api-key"); got != "sk-ant-config" {
		t.Errorf("upstream x-api-key=%q, want sk-ant-config", got)
	}
	if got := captured.headers.Get("Authorization"); got != "" {
		t.Errorf("Authorization should be stripped, got %q", got)
	}
}

func TestHandler_Gemini_CustomAuthHeader(t *testing.T) {
	upstream, captured := newUpstreamServer(t, `{"candidates":[]}`, 200)

	cfg := &config.Config{
		Models: []config.ModelConfig{
			{
				FrontendModelName: "my-gemini", BackendModelName: "gemini-1.5-pro", FrontendBaseURL: "/gemini",
				BackendBaseURL: upstream.URL, BackendAPIKey: "goog-config-token",
				BackendAuthHeader: "x-goog-api-key", BackendAuthSchema: "",
			},
		},
	}
	proxyServer := newProxyServer(t, cfg)

	reqBody := `{"model":"my-gemini","contents":[{"parts":[{"text":"hello"}]}]}`
	req, _ := http.NewRequest("POST", proxyServer.URL+"/gemini/v1/models/my-gemini:generateContent", strings.NewReader(reqBody))
	req.Header.Set("x-goog-api-key", "goog-client-token")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status=%d, want 200", resp.StatusCode)
	}
	if captured.model != "gemini-1.5-pro" {
		t.Errorf("upstream model=%q, want gemini-1.5-pro", captured.model)
	}
	if got := captured.headers.Get("x-goog-api-key"); got != "goog-config-token" {
		t.Errorf("upstream x-goog-api-key=%q, want goog-config-token", got)
	}
}

func TestHandler_ModelNotFound_Returns404(t *testing.T) {
	cfg := &config.Config{
		Models: []config.ModelConfig{
			{FrontendModelName: "other", BackendModelName: "other", FrontendBaseURL: "/openai", BackendBaseURL: "http://x"},
		},
	}
	proxyServer := newProxyServer(t, cfg)

	reqBody := `{"model":"nonexistent","messages":[]}`
	req, _ := http.NewRequest("POST", proxyServer.URL+"/openai/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status=%d, want 404", resp.StatusCode)
	}
}

func TestHandler_InvalidBody_Returns400(t *testing.T) {
	cfg := &config.Config{
		Models: []config.ModelConfig{
			{FrontendModelName: "x", BackendModelName: "x", FrontendBaseURL: "/openai", BackendBaseURL: "http://x"},
		},
	}
	proxyServer := newProxyServer(t, cfg)

	req, _ := http.NewRequest("POST", proxyServer.URL+"/openai/v1/chat/completions", strings.NewReader("not-json"))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status=%d, want 400", resp.StatusCode)
	}
}

func TestHandler_SSEStreaming(t *testing.T) {
	sseData := "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n"

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(sseData))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer upstream.Close()

	cfg := &config.Config{
		Models: []config.ModelConfig{
			{
				FrontendModelName: "my-gpt4", BackendModelName: "gpt-4", FrontendBaseURL: "/openai",
				BackendBaseURL: upstream.URL, BackendAPIKey: "sk-token",
				BackendAuthHeader: "Authorization", BackendAuthSchema: "Bearer",
			},
		},
	}
	proxyServer := newProxyServer(t, cfg)

	reqBody := `{"model":"my-gpt4","messages":[],"stream":true}`
	req, _ := http.NewRequest("POST", proxyServer.URL+"/openai/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status=%d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type=%q, want text/event-stream", ct)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if string(body) != sseData {
		t.Errorf("body mismatch\ngot:  %q\nwant: %q", string(body), sseData)
	}
}

func TestHandler_UpstreamErrorCode_IsForwarded(t *testing.T) {
	upstream, _ := newUpstreamServer(t, `{"error":{"message":"invalid model"}}`, 400)

	cfg := &config.Config{
		Models: []config.ModelConfig{
			{
				FrontendModelName: "my-gpt4", BackendModelName: "gpt-4", FrontendBaseURL: "/openai",
				BackendBaseURL: upstream.URL, BackendAPIKey: "sk-token",
				BackendAuthHeader: "Authorization", BackendAuthSchema: "Bearer",
			},
		},
	}
	proxyServer := newProxyServer(t, cfg)

	reqBody := `{"model":"my-gpt4","messages":[]}`
	req, _ := http.NewRequest("POST", proxyServer.URL+"/openai/v1/chat/completions", strings.NewReader(reqBody))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status=%d, want 400 (upstream error should be forwarded)", resp.StatusCode)
	}
}

// TestHandler_Translation_OAIChat_To_Claude sends an OpenAI Chat request through the proxy
// configured with incoming_dialect=openai-chat and upstream_dialect=claude.
// The upstream mock receives a Claude-format request and responds with a Claude response.
// The proxy must translate both directions.
func TestHandler_Translation_OAIChat_To_Claude(t *testing.T) {
	// Mock upstream: expects a Claude-format request, returns a Claude-format response.
	var capturedBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		// Return a valid Claude response.
		w.Write([]byte(`{
			"id": "msg_test",
			"type": "message",
			"role": "assistant",
			"model": "claude-3-5-sonnet-20241022",
			"content": [{"type": "text", "text": "Hello from Claude!"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`))
	}))
	defer upstream.Close()

	cfg := &config.Config{
		Models: []config.ModelConfig{
			{
				FrontendModelName:          "my-claude",
				BackendModelName:          "claude-3-5-sonnet-20241022",
				FrontendBaseURL: "/openai",
				BackendBaseURL: upstream.URL,
				BackendAPIKey:     "sk-ant-test",
				BackendAuthHeader:      "x-api-key",
				BackendAuthSchema:      "",
				FrontendDialect: "openai-chat",
				BackendDialect: "claude",
			},
		},
	}
	proxyServer := newProxyServer(t, cfg)

	// Client sends an OpenAI Chat request.
	reqBody := `{"model":"my-claude","messages":[{"role":"user","content":"Say hello"}]}`
	req, _ := http.NewRequest("POST", proxyServer.URL+"/openai/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-client")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status=%d, body=%s", resp.StatusCode, body)
	}

	// Verify upstream received a Claude-format request (has "max_tokens" and "content" array).
	var claudeReq map[string]interface{}
	if err := json.Unmarshal(capturedBody, &claudeReq); err != nil {
		t.Fatalf("upstream body is not valid JSON: %v", err)
	}
	if _, hasMaxTokens := claudeReq["max_tokens"]; !hasMaxTokens {
		t.Error("upstream request missing max_tokens (expected Claude format)")
	}
	msgs, _ := claudeReq["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Errorf("upstream messages count=%d, want 1", len(msgs))
	}
	// Claude messages use content as array of blocks.
	msg, _ := msgs[0].(map[string]interface{})
	if msg["role"] != "user" {
		t.Errorf("upstream message role=%q, want user", msg["role"])
	}

	// Verify response to client is in OpenAI Chat format (has "choices").
	respBody, _ := io.ReadAll(resp.Body)
	var oaiResp map[string]interface{}
	if err := json.Unmarshal(respBody, &oaiResp); err != nil {
		t.Fatalf("response body is not valid JSON: %v", err)
	}
	choices, _ := oaiResp["choices"].([]interface{})
	if len(choices) != 1 {
		t.Errorf("response choices count=%d, want 1", len(choices))
	}
	choice, _ := choices[0].(map[string]interface{})
	message, _ := choice["message"].(map[string]interface{})
	if message["content"] != "Hello from Claude!" {
		t.Errorf("response content=%q, want 'Hello from Claude!'", message["content"])
	}
}
