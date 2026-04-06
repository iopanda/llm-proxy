package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config holds the complete proxy configuration.
type Config struct {
	Server ServerConfig  `yaml:"server"`
	Models []ModelConfig `yaml:"models"`
}

// ServerConfig holds server-level settings.
type ServerConfig struct {
	Port int `yaml:"port"`
}

// ModelConfig defines a single model proxy mapping.
type ModelConfig struct {
	FrontendModelName string `yaml:"frontend_model_name"` // model name the client sends
	FrontendBaseURL   string `yaml:"frontend_base_url"`   // route prefix, must start with "/" (e.g., "/openai-compat")
	FrontendDialect   string `yaml:"frontend_dialect"`    // API dialect from the client (e.g., "openai-chat", "claude")

	BackendModelName string `yaml:"backend_model_name"` // actual model name forwarded to the upstream
	BackendBaseURL   string `yaml:"backend_base_url"`   // real upstream base URL (e.g., "https://api.openai.com")
	BackendAPIKey    string `yaml:"backend_api_key"`    // optional; empty means pass through client token
	BackendAuthHeader string `yaml:"backend_auth_header"` // header name for auth (e.g., "Authorization", "x-api-key")
	BackendAuthSchema string `yaml:"backend_auth_schema"` // token prefix (e.g., "Bearer"); empty means raw token
	BackendDialect   string `yaml:"backend_dialect"`    // API dialect for the upstream (e.g., "openai-responses", "bedrock")

	// System prompt modifications applied before forwarding to upstream.
	// SystemReplacements is a map of old→new text substitutions applied to the system field.
	SystemReplacements map[string]string `yaml:"system_replacements"`
	// SystemAppend is appended to the system field after replacements.
	SystemAppend string `yaml:"system_append"`
	// InjectProgress enables automatic injection of a session progress summary
	// (completed operations, inferred working directory, current task) into the system prompt.
	InjectProgress bool `yaml:"inject_progress"`
}

// Load reads and parses a YAML config file.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if cfg.Server.Port == 0 {
		cfg.Server.Port = 8080
	}

	return &cfg, nil
}

// UniqueFrontendBaseURLs returns the distinct frontend_base_url values across all model configs.
// Used to dynamically register routes at startup.
func (c *Config) UniqueFrontendBaseURLs() []string {
	seen := map[string]bool{}
	var baseURLs []string
	for _, m := range c.Models {
		if !seen[m.FrontendBaseURL] {
			seen[m.FrontendBaseURL] = true
			baseURLs = append(baseURLs, m.FrontendBaseURL)
		}
	}
	return baseURLs
}

// FindModel looks up a model config by frontend_base_url and frontend_model_name.
func (c *Config) FindModel(frontendBaseURL, modelName string) (*ModelConfig, bool) {
	for i := range c.Models {
		m := &c.Models[i]
		if m.FrontendBaseURL == frontendBaseURL && m.FrontendModelName == modelName {
			return m, true
		}
	}
	return nil, false
}
