package proxy

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/iopanda/llm-proxy/internal/config"
)

func TestParseModelName(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantModel string
		wantErr   bool
	}{
		{"valid", `{"model":"gpt-4","messages":[]}`, "gpt-4", false},
		{"missing model", `{"messages":[]}`, "", true},
		{"empty model", `{"model":""}`, "", true},
		{"invalid json", `{invalid}`, "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseModelName([]byte(tt.body))
			if (err != nil) != tt.wantErr {
				t.Errorf("error=%v, wantErr=%v", err, tt.wantErr)
			}
			if got != tt.wantModel {
				t.Errorf("model=%q, want %q", got, tt.wantModel)
			}
		})
	}
}

func TestRewriteModelName(t *testing.T) {
	body := `{"model":"my-gpt4","messages":[{"role":"user","content":"hello"}],"temperature":0.7}`
	result, err := rewriteModelName([]byte(body), "gpt-4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(result, &payload); err != nil {
		t.Fatalf("invalid JSON in result: %v", err)
	}
	if payload["model"] != "gpt-4" {
		t.Errorf("model=%q, want gpt-4", payload["model"])
	}
	if payload["messages"] == nil {
		t.Error("messages field should be preserved")
	}
	if payload["temperature"] != 0.7 {
		t.Errorf("temperature should be preserved, got %v", payload["temperature"])
	}
}

func TestSetAuthHeader_WithScheme(t *testing.T) {
	r, _ := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", nil)
	setAuthHeader(r, "Authorization", "Bearer", "sk-upstream")

	if got := r.Header.Get("Authorization"); got != "Bearer sk-upstream" {
		t.Errorf("Authorization=%q, want Bearer sk-upstream", got)
	}
}

func TestSetAuthHeader_WithoutScheme(t *testing.T) {
	r, _ := http.NewRequest(http.MethodPost, "https://api.anthropic.com/v1/messages", nil)
	setAuthHeader(r, "x-api-key", "", "sk-ant-upstream")

	if got := r.Header.Get("x-api-key"); got != "sk-ant-upstream" {
		t.Errorf("x-api-key=%q, want sk-ant-upstream", got)
	}
}

func TestSetAuthHeader_CustomHeader(t *testing.T) {
	r, _ := http.NewRequest(http.MethodPost, "https://generativelanguage.googleapis.com/v1beta/models", nil)
	setAuthHeader(r, "x-goog-api-key", "", "goog-token")

	if got := r.Header.Get("x-goog-api-key"); got != "goog-token" {
		t.Errorf("x-goog-api-key=%q, want goog-token", got)
	}
}

func TestBuildUpstreamRequest_UsesConfigToken(t *testing.T) {
	body := []byte(`{"model":"gpt-4-proxy","messages":[]}`)
	r := httptest.NewRequest(http.MethodPost, "/openai/v1/chat/completions", bytes.NewReader(body))
	r.Header.Set("Authorization", "Bearer sk-client")
	r.Header.Set("Content-Type", "application/json")

	modelCfg := &config.ModelConfig{
		FrontendModelName:          "gpt-4-proxy",
		BackendModelName:          "gpt-4",
		FrontendBaseURL: "/openai",
		BackendBaseURL: "https://api.openai.com",
		BackendAPIKey:     "sk-config",
		BackendAuthHeader:      "Authorization",
		BackendAuthSchema:      "Bearer",
	}

	req, err := buildUpstreamRequest(r, body, modelCfg, "/openai")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := req.Header.Get("Authorization"); got != "Bearer sk-config" {
		t.Errorf("Authorization=%q, want Bearer sk-config", got)
	}
	if req.URL.String() != "https://api.openai.com/v1/chat/completions" {
		t.Errorf("URL=%q, unexpected", req.URL.String())
	}
}

func TestBuildUpstreamRequest_PassthroughToken(t *testing.T) {
	body := []byte(`{"model":"gpt-4-proxy","messages":[]}`)
	r := httptest.NewRequest(http.MethodPost, "/openai/v1/chat/completions", bytes.NewReader(body))
	r.Header.Set("Authorization", "Bearer sk-client")

	modelCfg := &config.ModelConfig{
		FrontendModelName:          "gpt-4-proxy",
		BackendModelName:          "gpt-4",
		FrontendBaseURL: "/openai",
		BackendBaseURL: "https://api.openai.com",
		BackendAuthHeader:      "Authorization",
		BackendAuthSchema:      "Bearer",
		// no AccessToken — passthrough
	}

	req, err := buildUpstreamRequest(r, body, modelCfg, "/openai")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := req.Header.Get("Authorization"); got != "Bearer sk-client" {
		t.Errorf("Authorization=%q, want Bearer sk-client (passthrough)", got)
	}
}

func TestBuildUpstreamRequest_ConfigTokenStripsClientAuth(t *testing.T) {
	body := []byte(`{"model":"claude-proxy","messages":[]}`)
	r := httptest.NewRequest(http.MethodPost, "/claude/v1/messages", bytes.NewReader(body))
	r.Header.Set("x-api-key", "sk-ant-client")
	r.Header.Set("Authorization", "Bearer sk-also-client")

	modelCfg := &config.ModelConfig{
		FrontendModelName:          "claude-proxy",
		BackendModelName:          "claude-3-5-sonnet-20241022",
		FrontendBaseURL: "/claude",
		BackendBaseURL: "https://api.anthropic.com",
		BackendAPIKey:     "sk-ant-config",
		BackendAuthHeader:      "x-api-key",
		BackendAuthSchema:      "",
	}

	req, err := buildUpstreamRequest(r, body, modelCfg, "/claude")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := req.Header.Get("x-api-key"); got != "sk-ant-config" {
		t.Errorf("x-api-key=%q, want sk-ant-config", got)
	}
	if got := req.Header.Get("Authorization"); got != "" {
		t.Errorf("Authorization should be stripped, got %q", got)
	}
}

func TestBuildUpstreamRequest_RewritesModel_Gemini(t *testing.T) {
	body := []byte(`{"model":"my-gemini","contents":[]}`)
	r := httptest.NewRequest(http.MethodPost, "/gemini/v1beta/models/my-gemini:generateContent", bytes.NewReader(body))

	modelCfg := &config.ModelConfig{
		FrontendModelName:          "my-gemini",
		BackendModelName:          "gemini-1.5-pro",
		FrontendBaseURL: "/gemini",
		BackendBaseURL: "https://generativelanguage.googleapis.com",
		BackendAPIKey:     "goog-token",
		BackendAuthHeader:      "x-goog-api-key",
	}

	req, err := buildUpstreamRequest(r, body, modelCfg, "/gemini")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var payload map[string]interface{}
	if err := json.NewDecoder(req.Body).Decode(&payload); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if payload["model"] != "gemini-1.5-pro" {
		t.Errorf("model=%q, want gemini-1.5-pro", payload["model"])
	}
	if got := req.Header.Get("x-goog-api-key"); got != "goog-token" {
		t.Errorf("x-goog-api-key=%q, want goog-token", got)
	}
}
