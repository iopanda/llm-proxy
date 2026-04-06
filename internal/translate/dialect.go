package translate

import (
	"io"
	"net/http"

	"github.com/iopanda/llm-proxy/internal/translate/canonical"
)

// Dialect encapsulates encoding and decoding for a specific LLM API format.
type Dialect interface {
	// Name returns the dialect identifier (e.g., "openai-chat", "claude").
	Name() string

	// DecodeRequest parses a raw JSON request body into the canonical format.
	DecodeRequest(body []byte) (*canonical.Request, error)

	// EncodeRequest serializes a canonical request to this dialect's JSON format.
	// Extra headers specific to this dialect (e.g., anthropic-version) are returned separately.
	EncodeRequest(req *canonical.Request) (body []byte, extraHeaders http.Header, err error)

	// BuildUpstreamPath returns the URL path (and optional query string) to forward to the upstream.
	BuildUpstreamPath(incomingPath, mappedBaseURL, model string, stream bool) string

	// DecodeResponse parses a complete (non-streaming) JSON response body.
	DecodeResponse(body []byte) (*canonical.Response, error)

	// EncodeResponse serializes a canonical response to this dialect's JSON format.
	EncodeResponse(resp *canonical.Response, clientModel string) ([]byte, error)

	// StreamDecoder returns a decoder for reading streaming events from an upstream body.
	StreamDecoder(r io.Reader) StreamDecoder

	// StreamEncoder returns an encoder for writing streaming events to a client.
	StreamEncoder(w http.ResponseWriter, clientModel string) StreamEncoder
}

// StreamDecoder reads canonical events from an upstream streaming response.
type StreamDecoder interface {
	// Next returns the next event, or io.EOF when the stream ends.
	Next() (*canonical.StreamEvent, error)
}

// StreamEncoder writes canonical events to a downstream response in this dialect's format.
type StreamEncoder interface {
	Write(event *canonical.StreamEvent) error
	Flush() error
}
