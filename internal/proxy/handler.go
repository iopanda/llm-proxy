package proxy

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/iopanda/llm-proxy/internal/config"
	"github.com/iopanda/llm-proxy/internal/translate"
)

// Handler is the main HTTP proxy handler.
type Handler struct {
	config *config.Config
	client *http.Client
}

// NewHandler creates a new Handler with the given configuration.
func NewHandler(cfg *config.Config) *Handler {
	return &Handler{
		config: cfg,
		client: &http.Client{},
	}
}

// For returns an http.Handler for the given mapped_base_url group.
func (h *Handler) For(mappedBaseURL string) http.Handler {
	return h.handle(mappedBaseURL)
}

func (h *Handler) handle(mappedBaseURL string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		log.Printf("→ %s %s  remote=%s", r.Method, r.URL.Path, r.RemoteAddr)

		body, err := io.ReadAll(r.Body)
		r.Body.Close()
		if err != nil {
			log.Printf("  ERROR read body: %v", err)
			http.Error(w, "failed to read request body", http.StatusBadRequest)
			return
		}
		log.Printf("  body (%d bytes): %s", len(body), truncate(body, 512))

		modelName, err := parseModelName(body)
		if err != nil {
			log.Printf("  ERROR parse model name: %v", err)
			http.Error(w, "invalid request: "+err.Error(), http.StatusBadRequest)
			return
		}
		log.Printf("  model=%q  mapped_base_url=%q", modelName, mappedBaseURL)

		modelCfg, ok := h.config.FindModel(mappedBaseURL, modelName)
		if !ok {
			log.Printf("  ERROR model not found: %q (check config)", modelName)
			http.Error(w, "model not found: "+modelName, http.StatusNotFound)
			return
		}
		log.Printf("  matched config: source=%q  original_base_url=%q  access_token_set=%v",
			modelCfg.BackendModelName, modelCfg.BackendBaseURL, modelCfg.BackendAPIKey != "")

		if modelCfg.FrontendDialect != "" && modelCfg.BackendDialect != "" {
			log.Printf("  mode=translation  %s → %s", modelCfg.FrontendDialect, modelCfg.BackendDialect)
			h.handleTranslated(w, r, body, modelCfg, mappedBaseURL, start)
			return
		}

		log.Printf("  mode=proxy (passthrough)")
		upstreamReq, err := buildUpstreamRequest(r, body, modelCfg, mappedBaseURL)
		if err != nil {
			log.Printf("  ERROR build upstream request: %v", err)
			http.Error(w, "internal server error", http.StatusInternalServerError)
			return
		}
		log.Printf("  → upstream: %s %s", upstreamReq.Method, upstreamReq.URL)

		resp, err := h.client.Do(upstreamReq)
		if err != nil {
			log.Printf("  ERROR upstream request: %v", err)
			http.Error(w, "upstream error", http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		log.Printf("  ← upstream status=%d  elapsed=%s", resp.StatusCode, time.Since(start))
		copyResponseHeaders(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		streamResponse(w, resp.Body)
	})
}

func (h *Handler) handleTranslated(w http.ResponseWriter, r *http.Request, body []byte, modelCfg *config.ModelConfig, mappedBaseURL string, start time.Time) {
	inDialect, err := translate.Get(modelCfg.FrontendDialect)
	if err != nil {
		log.Printf("  ERROR unknown incoming dialect %q: %v", modelCfg.FrontendDialect, err)
		http.Error(w, "configuration error: "+err.Error(), http.StatusInternalServerError)
		return
	}
	upDialect, err := translate.Get(modelCfg.BackendDialect)
	if err != nil {
		log.Printf("  ERROR unknown upstream dialect %q: %v", modelCfg.BackendDialect, err)
		http.Error(w, "configuration error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	canReq, err := inDialect.DecodeRequest(body)
	if err != nil {
		log.Printf("  ERROR decode incoming request (%s): %v", modelCfg.FrontendDialect, err)
		http.Error(w, "invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	log.Printf("  canonical: messages=%d  stream=%v  tools=%d", len(canReq.Messages), canReq.Stream, len(canReq.Tools))
	canReq.Model = modelCfg.BackendModelName

	// Apply system prompt modifications (replacements, static append, progress injection).
	canReq.System = applySystemMods(canReq.System, modelCfg, canReq.Messages)

	upBody, extraHeaders, err := upDialect.EncodeRequest(canReq)
	if err != nil {
		log.Printf("  ERROR encode upstream request (%s): %v", modelCfg.BackendDialect, err)
		http.Error(w, "internal server error", http.StatusInternalServerError)
		return
	}

	upstreamPath := upDialect.BuildUpstreamPath(r.URL.Path, mappedBaseURL, modelCfg.BackendModelName, canReq.Stream)
	upstreamURL := strings.TrimRight(modelCfg.BackendBaseURL, "/") + upstreamPath
	log.Printf("  → upstream: %s %s", r.Method, upstreamURL)
	log.Printf("    body (%d bytes): %s", len(upBody), truncate(upBody, 512))

	upstreamReq, err := buildTranslatedRequest(r, upBody, modelCfg, upstreamURL)
	if err != nil {
		log.Printf("  ERROR build upstream request: %v", err)
		http.Error(w, "internal server error", http.StatusInternalServerError)
		return
	}
	for k, vs := range extraHeaders {
		for _, v := range vs {
			upstreamReq.Header.Set(k, v)
		}
	}
	logAuthHeaders(upstreamReq, modelCfg)

	resp, err := h.client.Do(upstreamReq)
	if err != nil {
		log.Printf("  ERROR upstream request failed: %v", err)
		http.Error(w, "upstream error", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	log.Printf("  ← upstream status=%d  elapsed=%s", resp.StatusCode, time.Since(start))

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		log.Printf("  upstream error body: %s", truncate(respBody, 512))
		copyResponseHeaders(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		w.Write(respBody)
		return
	}

	if canReq.Stream {
		log.Printf("  streaming response → client (dialect=%s)", modelCfg.FrontendDialect)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)

		decoder := upDialect.StreamDecoder(resp.Body)
		encoder := inDialect.StreamEncoder(w, modelCfg.FrontendModelName)
		eventCount := 0
		for {
			event, err := decoder.Next()
			if err == io.EOF {
				encoder.Flush()
				break
			}
			if err != nil {
				log.Printf("  ERROR decode upstream stream event #%d: %v", eventCount, err)
				break
			}
			log.Printf("  stream event #%d type=%s", eventCount, event.Type)
			if err := encoder.Write(event); err != nil {
				log.Printf("  ERROR encode downstream stream event #%d: %v", eventCount, err)
				break
			}
			eventCount++
		}
		log.Printf("  stream done: %d events  elapsed=%s", eventCount, time.Since(start))
		return
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("  ERROR read upstream response: %v", err)
		http.Error(w, "upstream read error", http.StatusBadGateway)
		return
	}
	log.Printf("  upstream response (%d bytes): %s", len(respBody), truncate(respBody, 512))

	canResp, err := upDialect.DecodeResponse(respBody)
	if err != nil {
		log.Printf("  ERROR decode upstream response (%s): %v — forwarding raw", modelCfg.BackendDialect, err)
		copyResponseHeaders(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		w.Write(respBody)
		return
	}
	outBody, err := inDialect.EncodeResponse(canResp, modelCfg.FrontendModelName)
	if err != nil {
		log.Printf("  ERROR encode downstream response (%s): %v", modelCfg.FrontendDialect, err)
		http.Error(w, "internal server error", http.StatusInternalServerError)
		return
	}
	log.Printf("  → client (%d bytes)  elapsed=%s", len(outBody), time.Since(start))

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	w.Write(outBody)
}

// logAuthHeaders logs which auth header is being sent upstream (value masked).
func logAuthHeaders(req *http.Request, modelCfg *config.ModelConfig) {
	if modelCfg.BackendAuthHeader == "" {
		return
	}
	val := req.Header.Get(modelCfg.BackendAuthHeader)
	if val == "" {
		log.Printf("    auth: header=%q NOT SET (check access_token or client token)", modelCfg.BackendAuthHeader)
	} else {
		masked := maskToken(val)
		log.Printf("    auth: %s: %s", modelCfg.BackendAuthHeader, masked)
	}
}

func maskToken(val string) string {
	if len(val) <= 8 {
		return "***"
	}
	return val[:4] + "..." + val[len(val)-4:]
}

// truncate returns up to n bytes of b as a printable string, with a suffix if clipped.
func truncate(b []byte, n int) string {
	b = bytes.Map(func(r rune) rune {
		if r == '\n' || r == '\r' {
			return ' '
		}
		return r
	}, b)
	if len(b) <= n {
		return fmt.Sprintf("%q", b)
	}
	return fmt.Sprintf("%q... (%d bytes total)", b[:n], len(b))
}

// streamResponse copies the response body to the writer, flushing after each chunk for SSE support.
func streamResponse(w http.ResponseWriter, body io.Reader) {
	flusher, canFlush := w.(http.Flusher)
	buf := make([]byte, 4096)
	for {
		n, err := body.Read(buf)
		if n > 0 {
			if _, writeErr := w.Write(buf[:n]); writeErr != nil {
				log.Printf("ERROR write response: %v", writeErr)
				return
			}
			if canFlush {
				flusher.Flush()
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Printf("ERROR read upstream body: %v", err)
			return
		}
	}
}

// copyResponseHeaders copies upstream response headers, skipping hop-by-hop headers.
func copyResponseHeaders(dst, src http.Header) {
	skip := map[string]bool{
		"Connection":        true,
		"Keep-Alive":        true,
		"Transfer-Encoding": true,
		"Upgrade":           true,
	}
	for key, values := range src {
		if skip[http.CanonicalHeaderKey(key)] {
			continue
		}
		for _, v := range values {
			dst.Add(key, v)
		}
	}
}
