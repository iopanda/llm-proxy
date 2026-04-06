package bedrock

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/iopanda/llm-proxy/internal/translate/canonical"
)

// -------- EncodeRequest --------

func TestEncodeRequest_SetsAnthropicVersion(t *testing.T) {
	d := &bedrockDialect{}
	req := &canonical.Request{
		Model:     "us.anthropic.claude-sonnet-4-6",
		MaxTokens: intPtr(512),
		Messages: []canonical.Message{
			{Role: canonical.RoleUser, Blocks: []canonical.Block{canonical.TextBlock{Text: "Hello"}}},
		},
	}

	body, headers, err := d.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	if headers != nil {
		t.Error("expected no extra headers for bedrock dialect")
	}

	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out["anthropic_version"] != "bedrock-2023-05-31" {
		t.Errorf("anthropic_version=%v, want bedrock-2023-05-31", out["anthropic_version"])
	}
	if _, hasModel := out["model"]; hasModel {
		t.Error("model field must not be present in bedrock request body")
	}
	if _, hasStream := out["stream"]; hasStream {
		t.Error("stream field must not be present in bedrock request body")
	}
}

func TestEncodeRequest_SystemAndMessages(t *testing.T) {
	d := &bedrockDialect{}
	req := &canonical.Request{
		System: "Be concise",
		Messages: []canonical.Message{
			{Role: canonical.RoleUser, Blocks: []canonical.Block{canonical.TextBlock{Text: "Hi"}}},
		},
	}

	body, _, err := d.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}

	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out["system"] != "Be concise" {
		t.Errorf("system=%v, want 'Be concise'", out["system"])
	}
	msgs, _ := out["messages"].([]any)
	if len(msgs) != 1 {
		t.Errorf("messages count=%d, want 1", len(msgs))
	}
}

func TestEncodeRequest_DefaultMaxTokens(t *testing.T) {
	d := &bedrockDialect{}
	req := &canonical.Request{
		Messages: []canonical.Message{
			{Role: canonical.RoleUser, Blocks: []canonical.Block{canonical.TextBlock{Text: "Hi"}}},
		},
	}

	body, _, err := d.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out["max_tokens"].(float64) != 1024 {
		t.Errorf("max_tokens=%v, want 1024", out["max_tokens"])
	}
}

// -------- BuildUpstreamPath --------

func TestBuildUpstreamPath_NonStreaming(t *testing.T) {
	d := &bedrockDialect{}
	path := d.BuildUpstreamPath("", "", "us.anthropic.claude-sonnet-4-6", false)
	want := "/model/us.anthropic.claude-sonnet-4-6/invoke"
	if path != want {
		t.Errorf("path=%q, want %q", path, want)
	}
}

func TestBuildUpstreamPath_Streaming(t *testing.T) {
	d := &bedrockDialect{}
	path := d.BuildUpstreamPath("", "", "us.anthropic.claude-sonnet-4-6", true)
	want := "/model/us.anthropic.claude-sonnet-4-6/invoke-with-response-stream"
	if path != want {
		t.Errorf("path=%q, want %q", path, want)
	}
}

func TestBuildUpstreamPath_ModelWithColon(t *testing.T) {
	d := &bedrockDialect{}
	path := d.BuildUpstreamPath("", "", "anthropic.claude-3-5-sonnet-20241022-v2:0", false)
	if !strings.Contains(path, "anthropic.claude-3-5-sonnet-20241022-v2:0") {
		t.Errorf("path=%q should contain raw model ID with colon", path)
	}
}

// -------- DecodeResponse --------

func TestDecodeResponse(t *testing.T) {
	d := &bedrockDialect{}
	body := `{
		"id": "msg_bdrk_abc",
		"model": "claude-sonnet-4-6",
		"content": [{"type": "text", "text": "Hello there"}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 12, "output_tokens": 5}
	}`

	resp, err := d.DecodeResponse([]byte(body))
	if err != nil {
		t.Fatalf("DecodeResponse: %v", err)
	}
	if resp.ID != "msg_bdrk_abc" {
		t.Errorf("ID=%q, want msg_bdrk_abc", resp.ID)
	}
	if resp.StopReason != canonical.StopReasonEndTurn {
		t.Errorf("StopReason=%q, want end_turn", resp.StopReason)
	}
	tb, ok := resp.Blocks[0].(canonical.TextBlock)
	if !ok || tb.Text != "Hello there" {
		t.Errorf("expected TextBlock{Hello there}, got %v", resp.Blocks[0])
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 5 {
		t.Errorf("Usage=%v, want {12,5}", resp.Usage)
	}
}

// -------- DecodeRequest --------

func TestDecodeRequest_NoModelField(t *testing.T) {
	d := &bedrockDialect{}
	body := `{
		"anthropic_version": "bedrock-2023-05-31",
		"max_tokens": 200,
		"system": "Be helpful",
		"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
	}`

	req, err := d.DecodeRequest([]byte(body))
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	if req.Model != "" {
		t.Errorf("Model=%q, want empty (model is in URL path for Bedrock)", req.Model)
	}
	if req.System != "Be helpful" {
		t.Errorf("System=%q, want 'Be helpful'", req.System)
	}
	if req.MaxTokens == nil || *req.MaxTokens != 200 {
		t.Errorf("MaxTokens=%v, want 200", req.MaxTokens)
	}
	if len(req.Messages) != 1 {
		t.Fatalf("messages count=%d, want 1", len(req.Messages))
	}
}

// -------- StreamDecoder --------

func TestStreamDecoder_ParsesTextDelta(t *testing.T) {
	d := &bedrockDialect{}

	// Build a sequence of EventStream frames representing a simple text response.
	events := []map[string]any{
		{"type": "message_start", "message": map[string]any{
			"id": "msg_1", "model": "claude-sonnet-4-6",
			"usage": map[string]any{"input_tokens": 5},
		}},
		{"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""}},
		{"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "text_delta", "text": "Hello"}},
		{"type": "content_block_stop", "index": 0},
		{"type": "message_delta",
			"delta": map[string]any{"stop_reason": "end_turn"},
			"usage": map[string]any{"output_tokens": 3}},
		{"type": "message_stop"},
	}

	r := buildEventStream(t, events)
	dec := d.StreamDecoder(r)

	var textDeltas []string
	var done bool
	for !done {
		event, err := dec.Next()
		if err == io.EOF {
			t.Fatal("unexpected EOF before message_stop")
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		switch event.Type {
		case canonical.EventTextDelta:
			textDeltas = append(textDeltas, event.TextDelta)
		case canonical.EventDone:
			done = true
			if event.StopReason != canonical.StopReasonEndTurn {
				t.Errorf("StopReason=%q, want end_turn", event.StopReason)
			}
			if event.Usage == nil || event.Usage.InputTokens != 5 || event.Usage.OutputTokens != 3 {
				t.Errorf("Usage=%v, want {5,3}", event.Usage)
			}
		}
	}

	if len(textDeltas) != 1 || textDeltas[0] != "Hello" {
		t.Errorf("textDeltas=%v, want [Hello]", textDeltas)
	}
}

func TestStreamDecoder_ParsesToolUse(t *testing.T) {
	d := &bedrockDialect{}

	events := []map[string]any{
		{"type": "message_start", "message": map[string]any{
			"usage": map[string]any{"input_tokens": 10},
		}},
		{"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "tool_use", "id": "tool_1", "name": "search"}},
		{"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": `{"q":`}},
		{"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": `"cats"}`}},
		{"type": "content_block_stop", "index": 0},
		{"type": "message_delta",
			"delta": map[string]any{"stop_reason": "tool_use"},
			"usage": map[string]any{"output_tokens": 8}},
		{"type": "message_stop"},
	}

	r := buildEventStream(t, events)
	dec := d.StreamDecoder(r)

	var toolStart *canonical.StreamEvent
	var argDeltas []string
	var done bool
	for !done {
		event, err := dec.Next()
		if err == io.EOF {
			t.Fatal("unexpected EOF")
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		switch event.Type {
		case canonical.EventToolUseStart:
			toolStart = event
		case canonical.EventToolArgsDelta:
			argDeltas = append(argDeltas, event.ToolArgsDelta)
		case canonical.EventDone:
			done = true
			if event.StopReason != canonical.StopReasonToolUse {
				t.Errorf("StopReason=%q, want tool_use", event.StopReason)
			}
		}
	}

	if toolStart == nil {
		t.Fatal("expected EventToolUseStart")
	}
	if toolStart.ToolUseID != "tool_1" || toolStart.ToolName != "search" {
		t.Errorf("tool start: id=%q name=%q", toolStart.ToolUseID, toolStart.ToolName)
	}
	if strings.Join(argDeltas, "") != `{"q":"cats"}` {
		t.Errorf("argDeltas=%v", argDeltas)
	}
}

func TestStreamDecoder_EOF(t *testing.T) {
	d := &bedrockDialect{}
	dec := d.StreamDecoder(strings.NewReader(""))
	_, err := dec.Next()
	if err != io.EOF {
		t.Errorf("expected io.EOF on empty stream, got %v", err)
	}
}

// -------- Cross-dialect: OAI Chat → Bedrock --------

func TestTranslate_OAIChat_To_Bedrock(t *testing.T) {
	// Simulate what the proxy handler does: decode openai-chat, encode bedrock.
	oaiBody := `{
		"model": "claude-via-openai",
		"messages": [
			{"role": "system", "content": "You are a pirate"},
			{"role": "user", "content": "Say hello"}
		],
		"max_tokens": 100,
		"stream": false
	}`

	// Decode incoming (openai-chat) to canonical — replicate decoder logic inline.
	var oaiReq struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		MaxTokens *int `json:"max_tokens"`
		Stream    bool `json:"stream"`
	}
	if err := json.Unmarshal([]byte(oaiBody), &oaiReq); err != nil {
		t.Fatalf("unmarshal oai: %v", err)
	}

	maxTok := 100
	canReq := &canonical.Request{
		Model:     "us.anthropic.claude-sonnet-4-6",
		MaxTokens: &maxTok,
		Stream:    oaiReq.Stream,
	}
	for _, m := range oaiReq.Messages {
		if m.Role == "system" {
			canReq.System = m.Content
			continue
		}
		role := canonical.RoleUser
		if m.Role == "assistant" {
			role = canonical.RoleAssistant
		}
		canReq.Messages = append(canReq.Messages, canonical.Message{
			Role:   role,
			Blocks: []canonical.Block{canonical.TextBlock{Text: m.Content}},
		})
	}

	// Encode to bedrock.
	d := &bedrockDialect{}
	body, headers, err := d.EncodeRequest(canReq)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	if headers != nil {
		t.Error("expected no extra headers")
	}

	var out map[string]any
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("unmarshal bedrock body: %v", err)
	}

	if out["anthropic_version"] != "bedrock-2023-05-31" {
		t.Errorf("anthropic_version=%v", out["anthropic_version"])
	}
	if _, hasModel := out["model"]; hasModel {
		t.Error("model must not be in bedrock body")
	}
	if out["system"] != "You are a pirate" {
		t.Errorf("system=%v", out["system"])
	}

	msgs, _ := out["messages"].([]any)
	if len(msgs) != 1 {
		t.Errorf("messages count=%d, want 1 (system extracted)", len(msgs))
	}

	// Check upstream path.
	path := d.BuildUpstreamPath("", "", "us.anthropic.claude-sonnet-4-6", false)
	if path != "/model/us.anthropic.claude-sonnet-4-6/invoke" {
		t.Errorf("path=%q", path)
	}
}

// -------- Helpers --------

func intPtr(n int) *int { return &n }

// buildEventStream encodes a slice of event maps into AWS EventStream binary format.
// Each event is base64-encoded in {"bytes":"..."} and wrapped in a minimal binary frame.
func buildEventStream(t *testing.T, events []map[string]any) io.Reader {
	t.Helper()
	var buf []byte
	for _, evt := range events {
		data, err := json.Marshal(evt)
		if err != nil {
			t.Fatalf("marshal event: %v", err)
		}
		encoded := base64.StdEncoding.EncodeToString(data)
		payload, err := json.Marshal(map[string]string{"bytes": encoded})
		if err != nil {
			t.Fatalf("marshal frame payload: %v", err)
		}
		buf = append(buf, makeEventStreamFrame(payload)...)
	}
	return strings.NewReader(string(buf))
}

// makeEventStreamFrame wraps payload in a minimal AWS EventStream binary frame.
// Header section is empty (0 bytes); CRC fields are zeroed (decoder skips validation).
func makeEventStreamFrame(payload []byte) []byte {
	headersLen := uint32(0)
	totalLen := uint32(12 + headersLen + uint32(len(payload)) + 4)

	frame := make([]byte, totalLen)
	binary.BigEndian.PutUint32(frame[0:4], totalLen)
	binary.BigEndian.PutUint32(frame[4:8], headersLen)
	// frame[8:12] = prelude CRC (zeroed)
	copy(frame[12:], payload)
	// frame[totalLen-4:] = message CRC (zeroed)
	return frame
}
