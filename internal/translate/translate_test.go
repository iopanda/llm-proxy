package translate_test

import (
	"encoding/json"
	"testing"

	"github.com/iopanda/llm-proxy/internal/translate"
	"github.com/iopanda/llm-proxy/internal/translate/canonical"

	// Register all dialects via init().
	_ "github.com/iopanda/llm-proxy/internal/translate/claude"
	_ "github.com/iopanda/llm-proxy/internal/translate/oaichat"
	_ "github.com/iopanda/llm-proxy/internal/translate/oairesponses"
)

// -------- OpenAI Chat → Canonical --------

func TestOAIChat_DecodeRequest_SimpleText(t *testing.T) {
	d := mustGet(t, "openai-chat")
	body := `{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}],"stream":false}`

	req, err := d.DecodeRequest([]byte(body))
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	if req.Model != "gpt-4o" {
		t.Errorf("Model=%q, want gpt-4o", req.Model)
	}
	if len(req.Messages) != 1 {
		t.Fatalf("messages count=%d, want 1", len(req.Messages))
	}
	msg := req.Messages[0]
	if msg.Role != canonical.RoleUser {
		t.Errorf("role=%q, want user", msg.Role)
	}
	tb, ok := msg.Blocks[0].(canonical.TextBlock)
	if !ok || tb.Text != "Hello" {
		t.Errorf("block=%v, want TextBlock{Hello}", msg.Blocks[0])
	}
}

func TestOAIChat_DecodeRequest_SystemMessage(t *testing.T) {
	d := mustGet(t, "openai-chat")
	body := `{"model":"gpt-4","messages":[{"role":"system","content":"You are helpful"},{"role":"user","content":"Hi"}]}`

	req, err := d.DecodeRequest([]byte(body))
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	if req.System != "You are helpful" {
		t.Errorf("System=%q, want 'You are helpful'", req.System)
	}
	if len(req.Messages) != 1 {
		t.Errorf("messages count=%d, want 1 (system extracted)", len(req.Messages))
	}
}

func TestOAIChat_DecodeRequest_ToolCall(t *testing.T) {
	d := mustGet(t, "openai-chat")
	body := `{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "Search for cats"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{\"query\":\"cats\"}"}}
			]},
			{"role": "tool", "tool_call_id": "call_1", "content": "Found 42 cats"}
		]
	}`

	req, err := d.DecodeRequest([]byte(body))
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	if len(req.Messages) != 3 {
		t.Fatalf("messages count=%d, want 3", len(req.Messages))
	}

	assistantMsg := req.Messages[1]
	tb, ok := assistantMsg.Blocks[0].(canonical.ToolUseBlock)
	if !ok {
		t.Fatalf("expected ToolUseBlock, got %T", assistantMsg.Blocks[0])
	}
	if tb.ID != "call_1" || tb.Name != "search" {
		t.Errorf("ToolUseBlock ID=%q Name=%q, want call_1/search", tb.ID, tb.Name)
	}

	// ToolName must be populated for Gemini compatibility.
	tr, ok := req.Messages[2].Blocks[0].(canonical.ToolResultBlock)
	if !ok {
		t.Fatalf("expected ToolResultBlock, got %T", req.Messages[2].Blocks[0])
	}
	if tr.ToolName != "search" {
		t.Errorf("ToolResultBlock.ToolName=%q, want search", tr.ToolName)
	}
}

func TestOAIChat_EncodeRequest_RoundTrip(t *testing.T) {
	d := mustGet(t, "openai-chat")
	original := `{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}],"stream":false}`

	req, err := d.DecodeRequest([]byte(original))
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	body, _, err := d.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	req2, err := d.DecodeRequest(body)
	if err != nil {
		t.Fatalf("re-decode: %v", err)
	}
	if req2.Model != req.Model {
		t.Errorf("Model mismatch: %q vs %q", req2.Model, req.Model)
	}
	if len(req2.Messages) != len(req.Messages) {
		t.Errorf("messages count mismatch: %d vs %d", len(req2.Messages), len(req.Messages))
	}
}

// -------- Claude --------

func TestClaude_DecodeRequest_SimpleText(t *testing.T) {
	d := mustGet(t, "claude")
	body := `{
		"model": "claude-3-5-sonnet-20241022",
		"max_tokens": 1024,
		"system": "Be helpful",
		"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
	}`

	req, err := d.DecodeRequest([]byte(body))
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	if req.System != "Be helpful" {
		t.Errorf("System=%q, want 'Be helpful'", req.System)
	}
	tb, ok := req.Messages[0].Blocks[0].(canonical.TextBlock)
	if !ok || tb.Text != "Hello" {
		t.Errorf("expected TextBlock{Hello}, got %v", req.Messages[0].Blocks[0])
	}
}

func TestClaude_EncodeRequest_SetsAnthropicVersion(t *testing.T) {
	d := mustGet(t, "claude")
	req := &canonical.Request{
		Model:     "claude-3-5-sonnet-20241022",
		MaxTokens: intPtr(512),
		Messages: []canonical.Message{
			{Role: canonical.RoleUser, Blocks: []canonical.Block{canonical.TextBlock{Text: "Hi"}}},
		},
	}

	_, headers, err := d.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest: %v", err)
	}
	if v := headers.Get("anthropic-version"); v == "" {
		t.Error("expected anthropic-version header to be set")
	}
}

func TestClaude_DecodeResponse(t *testing.T) {
	d := mustGet(t, "claude")
	body := `{
		"id": "msg_abc",
		"model": "claude-3-5-sonnet-20241022",
		"content": [{"type": "text", "text": "Hello there"}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`

	resp, err := d.DecodeResponse([]byte(body))
	if err != nil {
		t.Fatalf("DecodeResponse: %v", err)
	}
	if resp.ID != "msg_abc" {
		t.Errorf("ID=%q, want msg_abc", resp.ID)
	}
	if resp.StopReason != canonical.StopReasonEndTurn {
		t.Errorf("StopReason=%q, want end_turn", resp.StopReason)
	}
	tb, ok := resp.Blocks[0].(canonical.TextBlock)
	if !ok || tb.Text != "Hello there" {
		t.Errorf("expected TextBlock{Hello there}, got %v", resp.Blocks[0])
	}
	if resp.Usage == nil || resp.Usage.InputTokens != 10 {
		t.Errorf("Usage=%v, want InputTokens=10", resp.Usage)
	}
}

// -------- Cross-dialect: OAI Chat → Claude --------

func TestTranslate_OAIChat_To_Claude(t *testing.T) {
	inD := mustGet(t, "openai-chat")
	upD := mustGet(t, "claude")

	oaiBody := `{
		"model": "my-claude",
		"messages": [
			{"role": "system", "content": "You are a pirate"},
			{"role": "user", "content": "Say hello"}
		],
		"max_tokens": 100,
		"temperature": 0.5
	}`

	canReq, err := inD.DecodeRequest([]byte(oaiBody))
	if err != nil {
		t.Fatalf("decode OAI request: %v", err)
	}
	canReq.Model = "claude-3-5-sonnet-20241022"

	claudeBody, headers, err := upD.EncodeRequest(canReq)
	if err != nil {
		t.Fatalf("encode Claude request: %v", err)
	}
	if headers.Get("anthropic-version") == "" {
		t.Error("missing anthropic-version header")
	}

	var claudeReq map[string]interface{}
	if err := json.Unmarshal(claudeBody, &claudeReq); err != nil {
		t.Fatalf("parse Claude body: %v", err)
	}
	if claudeReq["system"] != "You are a pirate" {
		t.Errorf("system=%q, want 'You are a pirate'", claudeReq["system"])
	}
	msgs, _ := claudeReq["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Errorf("messages count=%d, want 1 (system moved to top-level)", len(msgs))
	}
}

func TestTranslate_Claude_To_OAIChat(t *testing.T) {
	inD := mustGet(t, "claude")
	upD := mustGet(t, "openai-chat")

	claudeBody := `{
		"model": "my-gpt4",
		"max_tokens": 1024,
		"system": "You are helpful",
		"messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
	}`

	canReq, err := inD.DecodeRequest([]byte(claudeBody))
	if err != nil {
		t.Fatalf("decode Claude request: %v", err)
	}
	canReq.Model = "gpt-4o"

	oaiBody, _, err := upD.EncodeRequest(canReq)
	if err != nil {
		t.Fatalf("encode OAI request: %v", err)
	}

	var oaiReq map[string]interface{}
	if err := json.Unmarshal(oaiBody, &oaiReq); err != nil {
		t.Fatalf("parse OAI body: %v", err)
	}

	msgs, _ := oaiReq["messages"].([]interface{})
	if len(msgs) < 2 {
		t.Fatalf("messages count=%d, want >=2", len(msgs))
	}
	firstMsg, _ := msgs[0].(map[string]interface{})
	if firstMsg["role"] != "system" {
		t.Errorf("first message role=%q, want system", firstMsg["role"])
	}
}

// -------- OpenAI Responses --------

func TestOAIResponses_DecodeRequest(t *testing.T) {
	d := mustGet(t, "openai-responses")
	body := `{
		"model": "gpt-4o",
		"input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
		"instructions": "Be helpful",
		"max_output_tokens": 500
	}`

	req, err := d.DecodeRequest([]byte(body))
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	if req.System != "Be helpful" {
		t.Errorf("System=%q, want 'Be helpful'", req.System)
	}
	if req.MaxTokens == nil || *req.MaxTokens != 500 {
		t.Errorf("MaxTokens=%v, want 500", req.MaxTokens)
	}
	if len(req.Messages) != 1 {
		t.Fatalf("messages count=%d, want 1", len(req.Messages))
	}
}

func TestOAIResponses_DecodeRequest_Reasoning(t *testing.T) {
	d := mustGet(t, "openai-responses")
	body := `{"model":"gpt-4o","input":[],"reasoning":{"effort":"high","summary":"auto"}}`

	req, err := d.DecodeRequest([]byte(body))
	if err != nil {
		t.Fatalf("DecodeRequest: %v", err)
	}
	if req.Thinking == nil || !req.Thinking.Enabled {
		t.Error("expected Thinking to be enabled")
	}
	if req.Thinking.BudgetTokens != 15000 {
		t.Errorf("BudgetTokens=%d, want 15000 for 'high' effort", req.Thinking.BudgetTokens)
	}
}

func TestOAIResponses_DecodeResponse(t *testing.T) {
	d := mustGet(t, "openai-responses")
	body := `{
		"id": "resp_abc",
		"model": "gpt-4o",
		"output": [
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi!"}]}
		],
		"status": "completed",
		"usage": {"input_tokens": 5, "output_tokens": 3}
	}`

	resp, err := d.DecodeResponse([]byte(body))
	if err != nil {
		t.Fatalf("DecodeResponse: %v", err)
	}
	if len(resp.Blocks) != 1 {
		t.Fatalf("blocks count=%d, want 1", len(resp.Blocks))
	}
	tb, ok := resp.Blocks[0].(canonical.TextBlock)
	if !ok || tb.Text != "Hi!" {
		t.Errorf("expected TextBlock{Hi!}, got %v", resp.Blocks[0])
	}
}

// -------- Helpers --------

func mustGet(t *testing.T, name string) translate.Dialect {
	t.Helper()
	d, err := translate.Get(name)
	if err != nil {
		t.Fatalf("Get(%q): %v", name, err)
	}
	return d
}

func intPtr(n int) *int { return &n }
