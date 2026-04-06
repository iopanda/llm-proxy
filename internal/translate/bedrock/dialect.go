// Package bedrock implements the AWS Bedrock InvokeModel API dialect for Claude models.
//
// Request differences from the native Claude API:
//   - Model goes in the URL path, not the request body.
//   - Streaming is determined by the endpoint path (/invoke vs /invoke-with-response-stream), not a body field.
//   - "anthropic_version" is a required field in the request body (value: "bedrock-2023-05-31").
//
// Response differences:
//   - Non-streaming responses are identical in structure to the Claude Messages API.
//   - Streaming responses use the AWS EventStream binary framing. Each frame's payload is
//     {"bytes": "<base64>"} where the base64-decoded value is a Claude SSE event JSON object.
package bedrock

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/iopanda/llm-proxy/internal/translate"
	"github.com/iopanda/llm-proxy/internal/translate/canonical"
)

func init() {
	translate.Register(&bedrockDialect{})
}

type bedrockDialect struct{}

func (d *bedrockDialect) Name() string { return "bedrock" }

// -------- JSON types --------

// bedrockRequest mirrors the Claude Messages API request but omits "model" and "stream"
// (model goes in the URL path; streaming is chosen by the endpoint path).
type bedrockRequest struct {
	AnthropicVersion string           `json:"anthropic_version"` // always "bedrock-2023-05-31"
	MaxTokens        int              `json:"max_tokens"`
	System           string           `json:"system,omitempty"`
	Messages         []bedrockMessage `json:"messages"`
	Tools            []bedrockTool    `json:"tools,omitempty"`
	ToolChoice       *bedrockChoice   `json:"tool_choice,omitempty"`
	Thinking         *bedrockThinking `json:"thinking,omitempty"`
	Temperature      *float64         `json:"temperature,omitempty"`
	TopP             *float64         `json:"top_p,omitempty"`
}

type bedrockMessage struct {
	Role    string         `json:"role"`
	Content []bedrockBlock `json:"content"`
}

type bedrockBlock struct {
	Type      string         `json:"type"`
	Text      string         `json:"text,omitempty"`
	Thinking  string         `json:"thinking,omitempty"`
	Signature string         `json:"signature,omitempty"`
	Source    *bedrockSource `json:"source,omitempty"`
	ID        string         `json:"id,omitempty"`
	Name      string         `json:"name,omitempty"`
	Input     any            `json:"input,omitempty"`
	ToolUseID string         `json:"tool_use_id,omitempty"`
	Content   []bedrockBlock `json:"content,omitempty"`
	IsError   bool           `json:"is_error,omitempty"`
}

type bedrockSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type bedrockTool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"input_schema"`
}

type bedrockChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type bedrockThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type bedrockResponse struct {
	ID         string         `json:"id"`
	Model      string         `json:"model"`
	Content    []bedrockBlock `json:"content"`
	StopReason string         `json:"stop_reason"`
	Usage      *bedrockUsage  `json:"usage,omitempty"`
}

type bedrockUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// -------- DecodeRequest --------

// DecodeRequest parses a Bedrock-format request body.
// Note: the model field is absent (it lives in the URL path on real Bedrock), so
// canonical.Request.Model will be empty and must be set by the caller.
func (d *bedrockDialect) DecodeRequest(body []byte) (*canonical.Request, error) {
	var req bedrockRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("bedrock decode request: %w", err)
	}

	canReq := &canonical.Request{
		System:      req.System,
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

	toolIDToName := buildToolIDMap(req.Messages)
	for _, msg := range req.Messages {
		role := canonical.RoleUser
		if msg.Role == "assistant" {
			role = canonical.RoleAssistant
		}
		canReq.Messages = append(canReq.Messages, canonical.Message{
			Role:   role,
			Blocks: decodeBlocks(msg.Content, toolIDToName),
		})
	}
	return canReq, nil
}

// -------- EncodeRequest --------

// EncodeRequest serializes a canonical request to Bedrock InvokeModel body format.
// The model is intentionally omitted — it belongs in the URL path (see BuildUpstreamPath).
// No extra HTTP headers are needed; authentication uses Authorization: Bearer.
func (d *bedrockDialect) EncodeRequest(req *canonical.Request) ([]byte, http.Header, error) {
	maxTokens := 1024
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}

	out := bedrockRequest{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        maxTokens,
		System:           req.System,
		Temperature:      req.Temperature,
		TopP:             req.TopP,
	}

	if req.Thinking != nil && req.Thinking.Enabled {
		out.Thinking = &bedrockThinking{
			Type:         "enabled",
			BudgetTokens: req.Thinking.BudgetTokens,
		}
	}
	if req.ToolChoice != nil {
		out.ToolChoice = encodeToolChoice(req.ToolChoice)
	}
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, bedrockTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.Parameters,
		})
	}
	for _, msg := range req.Messages {
		bm, err := encodeMessage(msg)
		if err != nil {
			return nil, nil, err
		}
		if bm != nil {
			out.Messages = append(out.Messages, *bm)
		}
	}

	body, err := json.Marshal(out)
	return body, nil, err
}

// -------- BuildUpstreamPath --------

// BuildUpstreamPath returns the Bedrock InvokeModel or InvokeModelWithResponseStream path.
func (d *bedrockDialect) BuildUpstreamPath(_, _, model string, stream bool) string {
	if stream {
		return "/model/" + model + "/invoke-with-response-stream"
	}
	return "/model/" + model + "/invoke"
}

// -------- DecodeResponse --------

// DecodeResponse parses a Bedrock non-streaming response (identical structure to Claude API).
func (d *bedrockDialect) DecodeResponse(body []byte) (*canonical.Response, error) {
	var resp bedrockResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("bedrock decode response: %w", err)
	}

	canResp := &canonical.Response{
		ID:         resp.ID,
		Model:      resp.Model,
		StopReason: mapStopReason(resp.StopReason),
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

// -------- EncodeResponse --------

// EncodeResponse serializes a canonical response back to Bedrock/Claude JSON format.
func (d *bedrockDialect) EncodeResponse(resp *canonical.Response, clientModel string) ([]byte, error) {
	var content []bedrockBlock
	for _, blk := range resp.Blocks {
		cb, err := encodeBlock(blk)
		if err != nil {
			return nil, err
		}
		if cb != nil {
			content = append(content, *cb)
		}
	}

	out := bedrockResponse{
		ID:         resp.ID,
		Model:      clientModel,
		Content:    content,
		StopReason: mapCanonicalStopReason(resp.StopReason),
	}
	if resp.Usage != nil {
		out.Usage = &bedrockUsage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
		}
	}
	return json.Marshal(out)
}

// -------- StreamDecoder --------

// StreamDecoder returns a decoder for AWS EventStream binary streaming responses.
// Each binary frame carries a JSON payload {"bytes":"<base64>"} whose decoded value
// is a Claude SSE event JSON object (e.g. {"type":"content_block_delta",...}).
func (d *bedrockDialect) StreamDecoder(r io.Reader) translate.StreamDecoder {
	return &bedrockStreamDecoder{r: r}
}

type bedrockStreamDecoder struct {
	r               io.Reader
	currentBlock    string
	currentToolID   string
	currentToolName string
	stopReason      canonical.StopReason
	inputTokens     int
	outputTokens    int
}

func (dec *bedrockStreamDecoder) Next() (*canonical.StreamEvent, error) {
	for {
		payload, err := readEventStreamFrame(dec.r)
		if err != nil {
			return nil, err
		}

		var frame struct {
			Bytes string `json:"bytes"`
		}
		if err := json.Unmarshal(payload, &frame); err != nil || frame.Bytes == "" {
			continue
		}

		eventData, err := base64.StdEncoding.DecodeString(frame.Bytes)
		if err != nil {
			continue
		}

		event, err := dec.parseEvent(eventData)
		if err != nil || event == nil {
			continue
		}
		return event, nil
	}
}

// parseEvent interprets a Claude SSE event JSON object (already decoded from base64).
func (dec *bedrockStreamDecoder) parseEvent(data []byte) (*canonical.StreamEvent, error) {
	var envelope struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, err
	}

	switch envelope.Type {
	case "message_start":
		var evt struct {
			Message struct {
				Usage struct {
					InputTokens int `json:"input_tokens"`
				} `json:"usage"`
			} `json:"message"`
		}
		if err := json.Unmarshal(data, &evt); err == nil {
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
		if err := json.Unmarshal(data, &evt); err != nil {
			return nil, nil
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
		if err := json.Unmarshal(data, &evt); err != nil {
			return nil, nil
		}
		switch evt.Delta.Type {
		case "text_delta":
			if evt.Delta.Text != "" {
				return &canonical.StreamEvent{Type: canonical.EventTextDelta, TextDelta: evt.Delta.Text}, nil
			}
		case "thinking_delta":
			if evt.Delta.Thinking != "" {
				return &canonical.StreamEvent{Type: canonical.EventThinkingDelta, ThinkingDelta: evt.Delta.Thinking}, nil
			}
		case "input_json_delta":
			if evt.Delta.PartialJSON != "" {
				return &canonical.StreamEvent{Type: canonical.EventToolArgsDelta, ToolArgsDelta: evt.Delta.PartialJSON}, nil
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
		if err := json.Unmarshal(data, &evt); err == nil {
			dec.stopReason = mapStopReason(evt.Delta.StopReason)
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

	return nil, nil
}

// readEventStreamFrame reads one AWS EventStream binary frame and returns the JSON payload.
//
// Frame layout (big-endian):
//
//	[4] total_length  — total bytes including this prelude and the trailing CRC
//	[4] headers_length
//	[4] prelude_CRC
//	[headers_length] headers (skipped)
//	[total_length - 12 - headers_length - 4] payload  (JSON)
//	[4] message_CRC
func readEventStreamFrame(r io.Reader) ([]byte, error) {
	var prelude [12]byte
	if _, err := io.ReadFull(r, prelude[:]); err != nil {
		if err == io.ErrUnexpectedEOF {
			return nil, io.EOF
		}
		return nil, err
	}

	totalLen := binary.BigEndian.Uint32(prelude[0:4])
	headersLen := binary.BigEndian.Uint32(prelude[4:8])

	// Minimum valid frame: 12 (prelude) + 0 (headers) + 0 (payload) + 4 (CRC) = 16
	if totalLen < 16 || headersLen > totalLen-16 {
		return nil, fmt.Errorf("bedrock: invalid EventStream frame totalLen=%d headersLen=%d", totalLen, headersLen)
	}

	rest := make([]byte, totalLen-12)
	if _, err := io.ReadFull(r, rest); err != nil {
		return nil, err
	}

	// payload sits between headers and the trailing 4-byte CRC
	payload := rest[headersLen : len(rest)-4]
	return payload, nil
}

// -------- StreamEncoder --------

// StreamEncoder returns a Claude-format SSE encoder for downstream clients that speak Bedrock.
// This reuses the same SSE event format as the Claude dialect.
func (d *bedrockDialect) StreamEncoder(w http.ResponseWriter, clientModel string) translate.StreamEncoder {
	flusher, _ := w.(http.Flusher)
	return &bedrockStreamEncoder{
		w:           w,
		clientModel: clientModel,
		id:          fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		flusher:     flusher,
	}
}

// bedrockStreamEncoder emits Claude SSE events (same wire format as the claude dialect).
// Used when incoming_dialect = "bedrock" and the downstream client expects Bedrock streaming.
type bedrockStreamEncoder struct {
	w                http.ResponseWriter
	clientModel      string
	id               string
	flusher          http.Flusher
	blockIndex       int
	currentBlockType string
	sentStart        bool
}

func (enc *bedrockStreamEncoder) Write(event *canonical.StreamEvent) error {
	if !enc.sentStart {
		enc.sentStart = true
		if err := enc.writeSSE("message_start", map[string]any{
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
		return enc.writeSSE("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": enc.blockIndex - 1,
			"delta": map[string]any{"type": "text_delta", "text": event.TextDelta},
		})

	case canonical.EventThinkingDelta:
		if err := enc.ensureBlock("thinking", map[string]any{"type": "thinking", "thinking": ""}); err != nil {
			return err
		}
		return enc.writeSSE("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": enc.blockIndex - 1,
			"delta": map[string]any{"type": "thinking_delta", "thinking": event.ThinkingDelta},
		})

	case canonical.EventToolUseStart:
		if err := enc.closeBlock(); err != nil {
			return err
		}
		if err := enc.writeSSE("content_block_start", map[string]any{
			"type": "content_block_start", "index": enc.blockIndex,
			"content_block": map[string]any{
				"type": "tool_use", "id": event.ToolUseID,
				"name": event.ToolName, "input": map[string]any{},
			},
		}); err != nil {
			return err
		}
		enc.currentBlockType = "tool_use"
		enc.blockIndex++

	case canonical.EventToolArgsDelta:
		return enc.writeSSE("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": enc.blockIndex - 1,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": event.ToolArgsDelta},
		})

	case canonical.EventDone:
		if err := enc.closeBlock(); err != nil {
			return err
		}
		var usage map[string]any
		if event.Usage != nil {
			usage = map[string]any{"output_tokens": event.Usage.OutputTokens}
		}
		if err := enc.writeSSE("message_delta", map[string]any{
			"type":  "message_delta",
			"delta": map[string]any{"stop_reason": mapCanonicalStopReason(event.StopReason), "stop_sequence": nil},
			"usage": usage,
		}); err != nil {
			return err
		}
		return enc.writeSSE("message_stop", map[string]any{"type": "message_stop"})
	}
	return nil
}

func (enc *bedrockStreamEncoder) Flush() error {
	if enc.flusher != nil {
		enc.flusher.Flush()
	}
	return nil
}

func (enc *bedrockStreamEncoder) ensureBlock(blockType string, blockData map[string]any) error {
	if enc.currentBlockType == blockType {
		return nil
	}
	if err := enc.closeBlock(); err != nil {
		return err
	}
	if err := enc.writeSSE("content_block_start", map[string]any{
		"type": "content_block_start", "index": enc.blockIndex,
		"content_block": blockData,
	}); err != nil {
		return err
	}
	enc.currentBlockType = blockType
	enc.blockIndex++
	return nil
}

func (enc *bedrockStreamEncoder) closeBlock() error {
	if enc.currentBlockType == "" {
		return nil
	}
	if err := enc.writeSSE("content_block_stop", map[string]any{
		"type": "content_block_stop", "index": enc.blockIndex - 1,
	}); err != nil {
		return err
	}
	enc.currentBlockType = ""
	return nil
}

func (enc *bedrockStreamEncoder) writeSSE(eventName string, data any) error {
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

// -------- Helpers --------

func buildToolIDMap(msgs []bedrockMessage) map[string]string {
	m := map[string]string{}
	for _, msg := range msgs {
		for _, cb := range msg.Content {
			if cb.Type == "tool_use" {
				m[cb.ID] = cb.Name
			}
		}
	}
	return m
}

func decodeBlocks(cbs []bedrockBlock, toolIDToName map[string]string) []canonical.Block {
	var blocks []canonical.Block
	for _, cb := range cbs {
		switch cb.Type {
		case "text":
			blocks = append(blocks, canonical.TextBlock{Text: cb.Text})
		case "thinking", "redacted_thinking":
			blocks = append(blocks, canonical.ThinkingBlock{Content: cb.Thinking, Signature: cb.Signature})
		case "image":
			if cb.Source != nil {
				if cb.Source.Type == "base64" {
					blocks = append(blocks, canonical.ImageBlock{MIMEType: cb.Source.MediaType, Data: cb.Source.Data})
				} else {
					blocks = append(blocks, canonical.ImageBlock{URL: cb.Source.URL})
				}
			}
		case "tool_use":
			blocks = append(blocks, canonical.ToolUseBlock{ID: cb.ID, Name: cb.Name, Input: cb.Input})
		case "tool_result":
			name := ""
			if toolIDToName != nil {
				name = toolIDToName[cb.ToolUseID]
			}
			blocks = append(blocks, canonical.ToolResultBlock{
				ToolUseID: cb.ToolUseID,
				ToolName:  name,
				Blocks:    decodeBlocks(cb.Content, toolIDToName),
				IsError:   cb.IsError,
			})
		}
	}
	return blocks
}

func encodeMessage(msg canonical.Message) (*bedrockMessage, error) {
	var role string
	switch msg.Role {
	case canonical.RoleUser:
		role = "user"
	case canonical.RoleAssistant:
		role = "assistant"
	default:
		return nil, nil
	}
	var blocks []bedrockBlock
	for _, blk := range msg.Blocks {
		cb, err := encodeBlock(blk)
		if err != nil {
			return nil, err
		}
		if cb != nil {
			blocks = append(blocks, *cb)
		}
	}
	return &bedrockMessage{Role: role, Content: blocks}, nil
}

func encodeBlock(blk canonical.Block) (*bedrockBlock, error) {
	switch b := blk.(type) {
	case canonical.TextBlock:
		return &bedrockBlock{Type: "text", Text: b.Text}, nil
	case canonical.ThinkingBlock:
		return &bedrockBlock{Type: "thinking", Thinking: b.Content, Signature: b.Signature}, nil
	case canonical.ImageBlock:
		src := &bedrockSource{}
		if b.Data != "" {
			src.Type = "base64"
			src.MediaType = b.MIMEType
			src.Data = b.Data
		} else {
			src.Type = "url"
			src.URL = b.URL
		}
		return &bedrockBlock{Type: "image", Source: src}, nil
	case canonical.ToolUseBlock:
		return &bedrockBlock{Type: "tool_use", ID: b.ID, Name: b.Name, Input: b.Input}, nil
	case canonical.ToolResultBlock:
		var content []bedrockBlock
		for _, inner := range b.Blocks {
			cb, err := encodeBlock(inner)
			if err != nil {
				return nil, err
			}
			if cb != nil {
				content = append(content, *cb)
			}
		}
		return &bedrockBlock{
			Type:      "tool_result",
			ToolUseID: b.ToolUseID,
			Content:   content,
			IsError:   b.IsError,
		}, nil
	}
	return nil, nil
}

func encodeToolChoice(tc *canonical.ToolChoice) *bedrockChoice {
	switch tc.Type {
	case "auto":
		return &bedrockChoice{Type: "auto"}
	case "any":
		return &bedrockChoice{Type: "any"}
	case "tool":
		return &bedrockChoice{Type: "tool", Name: tc.Name}
	default:
		return nil
	}
}

func mapStopReason(reason string) canonical.StopReason {
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

func mapCanonicalStopReason(r canonical.StopReason) string {
	switch r {
	case canonical.StopReasonMaxTokens:
		return "max_tokens"
	case canonical.StopReasonToolUse:
		return "tool_use"
	default:
		return "end_turn"
	}
}
