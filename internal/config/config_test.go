package config_test

import (
	"os"
	"testing"

	"github.com/iopanda/llm-proxy/internal/config"
)

const testYAML = `
server:
  port: 9090
models:
  - frontend_model_name: gpt-4-proxy
    backend_model_name: gpt-4
    frontend_base_url: /openai
    backend_base_url: https://api.openai.com
    backend_api_key: sk-test
    backend_auth_header: Authorization
    backend_auth_schema: Bearer
  - frontend_model_name: claude-proxy
    backend_model_name: claude-3-5-sonnet-20241022
    frontend_base_url: /claude
    backend_base_url: https://api.anthropic.com
    backend_auth_header: x-api-key
`

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	f, err := os.CreateTemp("", "config-*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Remove(f.Name()) })
	if _, err := f.WriteString(content); err != nil {
		t.Fatal(err)
	}
	f.Close()
	return f.Name()
}

func TestLoad(t *testing.T) {
	path := writeTemp(t, testYAML)
	cfg, err := config.Load(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.Server.Port != 9090 {
		t.Errorf("port=%d, want 9090", cfg.Server.Port)
	}
	if len(cfg.Models) != 2 {
		t.Fatalf("models count=%d, want 2", len(cfg.Models))
	}

	m0 := cfg.Models[0]
	if m0.FrontendModelName != "gpt-4-proxy" {
		t.Errorf("frontend_model_name=%q, want gpt-4-proxy", m0.FrontendModelName)
	}
	if m0.BackendModelName != "gpt-4" {
		t.Errorf("backend_model_name=%q, want gpt-4", m0.BackendModelName)
	}
	if m0.FrontendBaseURL != "/openai" {
		t.Errorf("frontend_base_url=%q, want /openai", m0.FrontendBaseURL)
	}
	if m0.BackendBaseURL != "https://api.openai.com" {
		t.Errorf("backend_base_url=%q, want https://api.openai.com", m0.BackendBaseURL)
	}
	if m0.BackendAPIKey != "sk-test" {
		t.Errorf("backend_api_key=%q, want sk-test", m0.BackendAPIKey)
	}
	if m0.BackendAuthHeader != "Authorization" {
		t.Errorf("backend_auth_header=%q, want Authorization", m0.BackendAuthHeader)
	}
	if m0.BackendAuthSchema != "Bearer" {
		t.Errorf("backend_auth_schema=%q, want Bearer", m0.BackendAuthSchema)
	}
	if cfg.Models[1].BackendAuthHeader != "x-api-key" {
		t.Errorf("backend_auth_header=%q, want x-api-key", cfg.Models[1].BackendAuthHeader)
	}
}

func TestLoadDefaultPort(t *testing.T) {
	path := writeTemp(t, "models: []\n")
	cfg, err := config.Load(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Server.Port != 8080 {
		t.Errorf("port=%d, want default 8080", cfg.Server.Port)
	}
}

func TestLoadMissingFile(t *testing.T) {
	_, err := config.Load("/nonexistent/path.yaml")
	if err == nil {
		t.Fatal("expected error for missing file, got nil")
	}
}

func TestFindModel(t *testing.T) {
	cfg := &config.Config{
		Models: []config.ModelConfig{
			{FrontendModelName: "gpt-4-proxy", BackendModelName: "gpt-4", FrontendBaseURL: "/openai"},
			{FrontendModelName: "claude-proxy", BackendModelName: "claude-3-5-sonnet-20241022", FrontendBaseURL: "/claude"},
		},
	}

	tests := []struct {
		frontendBaseURL string
		modelName       string
		wantOK          bool
		wantSrc         string
	}{
		{"/openai", "gpt-4-proxy", true, "gpt-4"},
		{"/claude", "claude-proxy", true, "claude-3-5-sonnet-20241022"},
		{"/openai", "nonexistent", false, ""},
		{"/claude", "gpt-4-proxy", false, ""},
	}

	for _, tt := range tests {
		m, ok := cfg.FindModel(tt.frontendBaseURL, tt.modelName)
		if ok != tt.wantOK {
			t.Errorf("FindModel(%q, %q): ok=%v, want %v", tt.frontendBaseURL, tt.modelName, ok, tt.wantOK)
			continue
		}
		if ok && m.BackendModelName != tt.wantSrc {
			t.Errorf("FindModel(%q, %q): backend_model_name=%q, want %q", tt.frontendBaseURL, tt.modelName, m.BackendModelName, tt.wantSrc)
		}
	}
}

func TestUniqueFrontendBaseURLs(t *testing.T) {
	cfg := &config.Config{
		Models: []config.ModelConfig{
			{FrontendBaseURL: "/openai"},
			{FrontendBaseURL: "/claude"},
			{FrontendBaseURL: "/openai"}, // duplicate
			{FrontendBaseURL: "/gemini"},
		},
	}
	got := cfg.UniqueFrontendBaseURLs()
	want := map[string]bool{"/openai": true, "/claude": true, "/gemini": true}
	if len(got) != len(want) {
		t.Fatalf("count=%d, want %d", len(got), len(want))
	}
	for _, p := range got {
		if !want[p] {
			t.Errorf("unexpected value %q", p)
		}
	}
}
