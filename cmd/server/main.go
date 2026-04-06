package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"

	"github.com/iopanda/llm-proxy/internal/config"
	"github.com/iopanda/llm-proxy/internal/proxy"

	// Register all translation dialects.
	_ "github.com/iopanda/llm-proxy/internal/translate/bedrock"
	_ "github.com/iopanda/llm-proxy/internal/translate/claude"
	_ "github.com/iopanda/llm-proxy/internal/translate/oaichat"
	_ "github.com/iopanda/llm-proxy/internal/translate/oairesponses"
)

func main() {
	configPath := flag.String("config", "config/config.yaml", "path to config file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatalf("failed to load config: %v", err)
	}

	// Print loaded model configs.
	log.Printf("loaded %d model(s):", len(cfg.Models))
	for _, m := range cfg.Models {
		if m.FrontendDialect != "" && m.BackendDialect != "" {
			log.Printf("  [translation] mapped=%q  source=%q  %s→%s  upstream=%s",
				m.FrontendModelName, m.BackendModelName, m.FrontendDialect, m.BackendDialect, m.BackendBaseURL)
		} else {
			log.Printf("  [proxy]       mapped=%q  source=%q  upstream=%s",
				m.FrontendModelName, m.BackendModelName, m.BackendBaseURL)
		}
	}

	handler := proxy.NewHandler(cfg)
	mux := http.NewServeMux()

	// Register one route per unique mapped_base_url found in config.
	for _, baseURL := range cfg.UniqueFrontendBaseURLs() {
		path := baseURL + "/"
		mux.Handle(path, handler.For(baseURL))
		log.Printf("registered route: %s", path)
	}

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("→ GET /healthz")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	addr := fmt.Sprintf(":%d", cfg.Server.Port)
	log.Printf("llm-proxy listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}
