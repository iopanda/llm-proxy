// Package claude implements the Anthropic Claude Messages API dialect.
package claude

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/iopanda/llm-proxy/internal/translate"
	"github.com/iopanda/llm-proxy/internal/translate/canonical"
)

func init() {
	translate.Register(&claudeDialect{})
}

type claudeDialect struct{}

func (d *claudeDialect) Name() string { return "claude" }

// -------- JSON types --------

type claudeRequest struct {
	Model       string            `json:"model"`
	MaxTokens   int               `json:"max_tokens"`
	System      json.RawMessage   `json:"system,omitempty"` // string or []{"type":"text","text":"..."}
	Messages    []claudeMessage   `json:"messages"`
	Tools       []claudeTool      `json:"tools,omitempty"`
	ToolChoice  *claudeToolChoice `json:"tool_choice,omitempty"`
	Thinking    *claudeThinking   `json:"thinking,omitempty"`
	Temperature *float64          `json:"temperature,omitempty"`
	TopP        *float64          `json:"top_p,omitempty"`
	Stream      bool              `json:"stream,omitempty"`
}

type claudeMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"` // string or []claudeBlock
}

type claudeBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	Signature string          `json:"signature,omitempty"`
	Data      string          `json:"data,omitempty"` // redacted_thinking opaque payload
	Source    *claudeSource   `json:"source,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     any             `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"` // string or []claudeBlock for tool_result
	IsError   bool            `json:"is_error,omitempty"`
}

type claudeSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type claudeTool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"input_schema"`
}

type claudeToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type claudeThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type claudeResponse struct {
	ID         string        `json:"id"`
	Model      string        `json:"model"`
	Content    []claudeBlock `json:"content"`
	StopReason string        `json:"stop_reason"`
	Usage      *claudeUsage  `json:"usage,omitempty"`
}

type claudeUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// -------- DecodeRequest --------

func (d *claudeDialect) DecodeRequest(body []byte) (*canonical.Request, error) {
	var req claudeRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("claude decode request: %w", err)
	}

	canReq := &canonical.Request{
		Model:       req.Model,
		System:      parseSystemField(req.System),
		Stream:      req.Stream,
		Temperature: req.Temperature,
		TopP:        req.TopP,
	}
	if req.MaxTokens > 0 {
		canReq.MaxTokens = &req.MaxTokens
	}
	if req.Thinking != nil && req.Thinking.Type == "enabled" {
		canReq.Thinking = &canonical.ThinkingConfig{
			Enabled:      true,
			BudgetTokens: req.Thinking.BudgetTokens,
		}
	}
	if req.ToolChoice != nil {
		canReq.ToolChoice = &canonical.ToolChoice{
			Type: req.ToolChoice.Type,
			Name: req.ToolChoice.Name,
		}
	}
	for _, t := range req.Tools {
		canReq.Tools = append(canReq.Tools, canonical.Tool{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.InputSchema,
		})
	}

	// Build tool_use_id → tool_name map for ToolResultBlock resolution.
	toolIDToName := map[string]string{}
	for _, msg := range req.Messages {
		blocks, _ := parseContentField(msg.Content)
		for _, cb := range blocks {
			if cb.Type == "tool_use" {
				toolIDToName[cb.ID] = cb.Name
			}
		}
	}

	for _, msg := range req.Messages {
		role := canonical.RoleUser
		if msg.Role == "assistant" {
			role = canonical.RoleAssistant
		}
		blocks, _ := parseContentField(msg.Content)
		canReq.Messages = append(canReq.Messages, canonical.Message{
			Role:   role,
			Blocks: decodeBlocks(blocks, toolIDToName),
		})
	}
	return canReq, nil
}

func decodeBlocks(cbs []claudeBlock, toolIDToName map[string]string) []canonical.Block {
	var blocks []canonical.Block
	for _, cb := range cbs {
		switch cb.Type {
		case "text":
			blocks = append(blocks, canonical.TextBlock{Text: cb.Text})
		case "thinking":
			blocks = append(blocks, canonical.ThinkingBlock{
				Content:   cb.Thinking,
				Signature: cb.Signature,
			})
		case "redacted_thinking":
			blocks = append(blocks, canonical.ThinkingBlock{
				Signature: cb.Data, // redacted_thinking uses "data" field, not "signature"
			})
		case "image":
			if cb.Source != nil {
				if cb.Source.Type == "base64" {
					blocks = append(blocks, canonical.ImageBlock{
						MIMEType: cb.Source.MediaType,
						Data:     cb.Source.Data,
					})
				} else {
					blocks = append(blocks, canonical.ImageBlock{URL: cb.Source.URL})
				}
			}
		case "tool_use":
			blocks = append(blocks, canonical.ToolUseBlock{
				ID:    cb.ID,
				Name:  cb.Name,
				Input: cb.Input,
			})
		case "tool_result":
			name := ""
			if toolIDToName != nil {
				name = toolIDToName[cb.ToolUseID]
			}
			innerBlocks, _ := parseContentField(cb.Content)
			blocks = append(blocks, canonical.ToolResultBlock{
				ToolUseID: cb.ToolUseID,
				ToolName:  name,
				Blocks:    decodeBlocks(innerBlocks, toolIDToName),
				IsError:   cb.IsError,
			})
		}
	}
	return blocks
}

// -------- EncodeRequest --------

func (d *claudeDialect) EncodeRequest(req *canonical.Request) ([]byte, http.Header, error) {
	maxTokens := 1024
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}

	out := claudeRequest{
		Model:       req.Model,
		MaxTokens:   maxTokens,
		System:      marshalSystemField(req.System),
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
	}

	if req.Thinking != nil && req.Thinking.Enabled {
		out.Thinking = &claudeThinking{
			Type:         "enabled",
			BudgetTokens: req.Thinking.BudgetTokens,
		}
	}
	if req.ToolChoice != nil {
		out.ToolChoice = encodeToolChoice(req.ToolChoice)
	}
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, claudeTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.Parameters,
		})
	}

	for _, msg := range req.Messages {
		cm, err := encodeMessage(msg)
		if err != nil {
			return nil, nil, err
		}
		if cm != nil {
			out.Messages = append(out.Messages, *cm)
		}
	}

	headers := http.Header{}
	headers.Set("anthropic-version", "2023-06-01")
	body, err := json.Marshal(out)
	return body, headers, err
}

func encodeMessage(msg canonical.Message) (*claudeMessage, error) {
	var role string
	switch msg.Role {
	case canonical.RoleUser:
		role = "user"
	case canonical.RoleAssistant:
		role = "assistant"
	default:
		return nil, nil
	}

	var blocks []claudeBlock
	for _, blk := range msg.Blocks {
		cb, err := encodeBlock(blk)
		if err != nil {
			return nil, err
		}
		if cb != nil {
			blocks = append(blocks, *cb)
		}
	}
	contentJSON, err := json.Marshal(blocks)
	if err != nil {
		return nil, err
	}
	return &claudeMessage{Role: role, Content: contentJSON}, nil
}

func encodeBlock(blk canonical.Block) (*claudeBlock, error) {
	switch b := blk.(type) {
	case canonical.TextBlock:
		return &claudeBlock{Type: "text", Text: b.Text}, nil
	case canonical.ThinkingBlock:
		if b.Content == "" && b.Signature != "" {
			// Opaque signature (e.g., Gemini's thoughtSignature) — encode as redacted_thinking.
			return &claudeBlock{Type: "redacted_thinking", Data: b.Signature}, nil
		}
		return &claudeBlock{
			Type:      "thinking",
			Thinking:  b.Content,
			Signature: b.Signature,
		}, nil
	case canonical.ImageBlock:
		src := &claudeSource{}
		if b.Data != "" {
			src.Type = "base64"
			src.MediaType = b.MIMEType
			src.Data = b.Data
		} else {
			src.Type = "url"
			src.URL = b.URL
		}
		return &claudeBlock{Type: "image", Source: src}, nil
	case canonical.ToolUseBlock:
		return &claudeBlock{
			Type:  "tool_use",
			ID:    b.ID,
			Name:  b.Name,
			Input: b.Input,
		}, nil
	case canonical.ToolResultBlock:
		var innerBlocks []claudeBlock
		for _, inner := range b.Blocks {
			cb, err := encodeBlock(inner)
			if err != nil {
				return nil, err
			}
			if cb != nil {
				innerBlocks = append(innerBlocks, *cb)
			}
		}
		contentJSON, err := json.Marshal(innerBlocks)
		if err != nil {
			return nil, err
		}
		return &claudeBlock{
			Type:      "tool_result",
			ToolUseID: b.ToolUseID,
			Content:   contentJSON,
			IsError:   b.IsError,
		}, nil
	}
	return nil, nil
}

func encodeToolChoice(tc *canonical.ToolChoice) *claudeToolChoice {
	switch tc.Type {
	case "auto":
		return &claudeToolChoice{Type: "auto"}
	case "any":
		return &claudeToolChoice{Type: "any"}
	case "tool":
		return &claudeToolChoice{Type: "tool", Name: tc.Name}
	default:
		return nil
	}
}

// -------- BuildUpstreamPath --------

func (d *claudeDialect) BuildUpstreamPath(incomingPath, mappedBaseURL, model string, stream bool) string {
	return "/messages"
}

// -------- DecodeResponse --------

func (d *claudeDialect) DecodeResponse(body []byte) (*canonical.Response, error) {
	var resp claudeResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("claude decode response: %w", err)
	}

	canResp := &canonical.Response{
		ID:         resp.ID,
		Model:      resp.Model,
		StopReason: mapClaudeStopReason(resp.StopReason),
		Blocks:     decodeBlocks(resp.Content, nil),
	}
	if resp.Usage != nil {
		canResp.Usage = &canonical.TokenUsage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
		}
	}
	return canResp, nil
}

func mapClaudeStopReason(reason string) canonical.StopReason {
	switch reason {
	case "end_turn":
		return canonical.StopReasonEndTurn
	case "max_tokens":
		return canonical.StopReasonMaxTokens
	case "tool_use":
		return canonical.StopReasonToolUse
	default:
		return canonical.StopReasonEndTurn
	}
}

func mapCanonicalToClaudeStopReason(reason canonical.StopReason) string {
	switch reason {
	case canonical.StopReasonMaxTokens:
		return "max_tokens"
	case canonical.StopReasonToolUse:
		return "tool_use"
	default:
		return "end_turn"
	}
}

// -------- EncodeResponse --------

func (d *claudeDialect) EncodeResponse(resp *canonical.Response, clientModel string) ([]byte, error) {
	var content []claudeBlock
	for _, blk := range resp.Blocks {
		cb, err := encodeBlock(blk)
		if err != nil {
			return nil, err
		}
		if cb != nil {
			content = append(content, *cb)
		}
	}

	out := claudeResponse{
		ID:         resp.ID,
		Model:      clientModel,
		Content:    content,
		StopReason: mapCanonicalToClaudeStopReason(resp.StopReason),
	}
	if resp.Usage != nil {
		out.Usage = &claudeUsage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
		}
	}
	return json.Marshal(out)
}

// -------- StreamDecoder --------

type claudeStreamDecoder struct {
	scanner         *bufio.Scanner
	pendingEvent    string
	currentBlock    string // "text", "thinking", "tool_use"
	currentToolID   string
	currentToolName string
	stopReason      canonical.StopReason
	inputTokens     int
	outputTokens    int
}

func (d *claudeDialect) StreamDecoder(r io.Reader) translate.StreamDecoder {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)
	return &claudeStreamDecoder{scanner: sc}
}

func (dec *claudeStreamDecoder) Next() (*canonical.StreamEvent, error) {
	for {
		if !dec.scanner.Scan() {
			if err := dec.scanner.Err(); err != nil {
				return nil, err
			}
			return nil, io.EOF
		}
		line := dec.scanner.Text()

		if strings.HasPrefix(line, "event: ") {
			dec.pendingEvent = strings.TrimPrefix(line, "event: ")
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		switch dec.pendingEvent {
		case "message_start":
			var evt struct {
				Message struct {
					Usage struct {
						InputTokens int `json:"input_tokens"`
					} `json:"usage"`
				} `json:"message"`
			}
			if err := json.Unmarshal([]byte(data), &evt); err == nil {
				dec.inputTokens = evt.Message.Usage.InputTokens
			}

		case "content_block_start":
			var evt struct {
				ContentBlock struct {
					Type string `json:"type"`
					ID   string `json:"id"`
					Name string `json:"name"`
				} `json:"content_block"`
			}
			if err := json.Unmarshal([]byte(data), &evt); err != nil {
				continue
			}
			dec.currentBlock = evt.ContentBlock.Type
			if dec.currentBlock == "tool_use" {
				dec.currentToolID = evt.ContentBlock.ID
				dec.currentToolName = evt.ContentBlock.Name
				return &canonical.StreamEvent{
					Type:      canonical.EventToolUseStart,
					ToolUseID: dec.currentToolID,
					ToolName:  dec.currentToolName,
				}, nil
			}

		case "content_block_delta":
			var evt struct {
				Delta struct {
					Type        string `json:"type"`
					Text        string `json:"text"`
					Thinking    string `json:"thinking"`
					PartialJSON string `json:"partial_json"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &evt); err != nil {
				continue
			}
			switch evt.Delta.Type {
			case "text_delta":
				if evt.Delta.Text != "" {
					return &canonical.StreamEvent{
						Type:      canonical.EventTextDelta,
						TextDelta: evt.Delta.Text,
					}, nil
				}
			case "thinking_delta":
				if evt.Delta.Thinking != "" {
					return &canonical.StreamEvent{
						Type:          canonical.EventThinkingDelta,
						ThinkingDelta: evt.Delta.Thinking,
					}, nil
				}
			case "input_json_delta":
				if evt.Delta.PartialJSON != "" {
					return &canonical.StreamEvent{
						Type:          canonical.EventToolArgsDelta,
						ToolArgsDelta: evt.Delta.PartialJSON,
					}, nil
				}
			}

		case "message_delta":
			var evt struct {
				Delta struct {
					StopReason string `json:"stop_reason"`
				} `json:"delta"`
				Usage struct {
					OutputTokens int `json:"output_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal([]byte(data), &evt); err == nil {
				dec.stopReason = mapClaudeStopReason(evt.Delta.StopReason)
				dec.outputTokens = evt.Usage.OutputTokens
			}

		case "message_stop":
			return &canonical.StreamEvent{
				Type:       canonical.EventDone,
				StopReason: dec.stopReason,
				Usage: &canonical.TokenUsage{
					InputTokens:  dec.inputTokens,
					OutputTokens: dec.outputTokens,
				},
			}, nil
		}
	}
}

// -------- StreamEncoder --------

type claudeStreamEncoder struct {
	w                http.ResponseWriter
	clientModel      string
	id               string
	flusher          http.Flusher
	blockIndex       int
	currentBlockType string
	sentStart        bool
}

func (d *claudeDialect) StreamEncoder(w http.ResponseWriter, clientModel string) translate.StreamEncoder {
	flusher, _ := w.(http.Flusher)
	return &claudeStreamEncoder{
		w:           w,
		clientModel: clientModel,
		id:          fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		flusher:     flusher,
	}
}

func (enc *claudeStreamEncoder) Write(event *canonical.StreamEvent) error {
	if !enc.sentStart {
		enc.sentStart = true
		if err := enc.writeEvent("message_start", map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id":            enc.id,
				"type":          "message",
				"role":          "assistant",
				"model":         enc.clientModel,
				"content":       []any{},
				"stop_reason":   nil,
				"stop_sequence": nil,
				"usage":         map[string]any{"input_tokens": 0, "output_tokens": 1},
			},
		}); err != nil {
			return err
		}
	}

	switch event.Type {
	case canonical.EventTextDelta:
		if err := enc.ensureBlock("text", map[string]any{"type": "text", "text": ""}); err != nil {
			return err
		}
		return enc.writeEvent("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": enc.blockIndex - 1,
			"delta": map[string]any{"type": "text_delta", "text": event.TextDelta},
		})

	case canonical.EventThinkingDelta:
		if err := enc.ensureBlock("thinking", map[string]any{"type": "thinking", "thinking": ""}); err != nil {
			return err
		}
		return enc.writeEvent("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": enc.blockIndex - 1,
			"delta": map[string]any{"type": "thinking_delta", "thinking": event.ThinkingDelta},
		})

	case canonical.EventToolUseStart:
		if err := enc.closeCurrentBlock(); err != nil {
			return err
		}
		// If the tool call carries a thought signature (e.g., Gemini), emit a
		// redacted_thinking block first so Claude Code will preserve and return it.
		if event.ThoughtSignature != "" {
			if err := enc.writeEvent("content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": enc.blockIndex,
				"content_block": map[string]any{
					"type": "redacted_thinking",
					"data": event.ThoughtSignature,
				},
			}); err != nil {
				return err
			}
			if err := enc.writeEvent("content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": enc.blockIndex,
			}); err != nil {
				return err
			}
			enc.blockIndex++
		}
		if err := enc.writeEvent("content_block_start", map[string]any{
			"type":  "content_block_start",
			"index": enc.blockIndex,
			"content_block": map[string]any{
				"type":  "tool_use",
				"id":    event.ToolUseID,
				"name":  event.ToolName,
				"input": map[string]any{},
			},
		}); err != nil {
			return err
		}
		enc.currentBlockType = "tool_use"
		enc.blockIndex++

	case canonical.EventToolArgsDelta:
		return enc.writeEvent("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": enc.blockIndex - 1,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": event.ToolArgsDelta},
		})

	case canonical.EventDone:
		if err := enc.closeCurrentBlock(); err != nil {
			return err
		}
		var usageMap map[string]any
		if event.Usage != nil {
			usageMap = map[string]any{"output_tokens": event.Usage.OutputTokens}
		}
		if err := enc.writeEvent("message_delta", map[string]any{
			"type":  "message_delta",
			"delta": map[string]any{"stop_reason": mapCanonicalToClaudeStopReason(event.StopReason), "stop_sequence": nil},
			"usage": usageMap,
		}); err != nil {
			return err
		}
		return enc.writeEvent("message_stop", map[string]any{"type": "message_stop"})
	}
	return nil
}

func (enc *claudeStreamEncoder) ensureBlock(blockType string, blockData map[string]any) error {
	if enc.currentBlockType == blockType {
		return nil
	}
	if err := enc.closeCurrentBlock(); err != nil {
		return err
	}
	if err := enc.writeEvent("content_block_start", map[string]any{
		"type":          "content_block_start",
		"index":         enc.blockIndex,
		"content_block": blockData,
	}); err != nil {
		return err
	}
	enc.currentBlockType = blockType
	enc.blockIndex++
	return nil
}

func (enc *claudeStreamEncoder) closeCurrentBlock() error {
	if enc.currentBlockType == "" {
		return nil
	}
	if err := enc.writeEvent("content_block_stop", map[string]any{
		"type":  "content_block_stop",
		"index": enc.blockIndex - 1,
	}); err != nil {
		return err
	}
	enc.currentBlockType = ""
	return nil
}

func (enc *claudeStreamEncoder) writeEvent(eventName string, data any) error {
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(enc.w, "event: %s\ndata: %s\n\n", eventName, b)
	if enc.flusher != nil {
		enc.flusher.Flush()
	}
	return err
}

func (enc *claudeStreamEncoder) Flush() error {
	if enc.flusher != nil {
		enc.flusher.Flush()
	}
	return nil
}

// parseContentField decodes the Claude API "content" field, which may be either a plain
// JSON string or an array of content blocks. A string is converted to a single text block.
func parseContentField(raw json.RawMessage) ([]claudeBlock, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	// Array form: [{"type":"text","text":"..."}]
	var blocks []claudeBlock
	if err := json.Unmarshal(raw, &blocks); err == nil {
		return blocks, nil
	}
	// Plain string form: "Hello"
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, err
	}
	return []claudeBlock{{Type: "text", Text: s}}, nil
}

// parseSystemField decodes the Claude API "system" field, which may be either a plain
// JSON string or an array of content blocks: [{"type":"text","text":"..."},...].
// All text blocks are concatenated with a newline separator.
func parseSystemField(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	// Plain string form: "Be helpful"
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	// Array form: [{"type":"text","text":"Be helpful"}]
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return ""
	}
	var parts []string
	for _, b := range blocks {
		if b.Type == "text" && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// marshalSystemField encodes a canonical system string back to a JSON string value
// suitable for the claudeRequest.System field (json.RawMessage).
func marshalSystemField(system string) json.RawMessage {
	if system == "" {
		return nil
	}
	b, _ := json.Marshal(system)
	return b
}
