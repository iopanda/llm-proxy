// Package oairesponses implements the OpenAI Responses API dialect (used by Codex CLI 2025).
package oairesponses

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
	translate.Register(&oaiRespDialect{})
}

type oaiRespDialect struct{}

func (d *oaiRespDialect) Name() string { return "openai-responses" }

// -------- JSON types --------

type oaiRespRequest struct {
	Model           string        `json:"model"`
	Input           []interface{} `json:"input"`
	Instructions    string        `json:"instructions,omitempty"`
	MaxOutputTokens *int          `json:"max_output_tokens,omitempty"`
	Temperature     *float64      `json:"temperature,omitempty"`
	TopP            *float64      `json:"top_p,omitempty"`
	Stream          bool          `json:"stream,omitempty"`
	Tools           []oaiRespTool `json:"tools,omitempty"`
	Reasoning       *oaiRespReasoningCfg `json:"reasoning,omitempty"`
}

type oaiRespTool struct {
	Type        string `json:"type"` // "function"
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type oaiRespReasoningCfg struct {
	Effort  string `json:"effort,omitempty"`  // "low", "medium", "high"
	Summary string `json:"summary,omitempty"` // "auto", "concise", "detailed"
}

// Response types

type oaiRespResponse struct {
	ID     string           `json:"id"`
	Model  string           `json:"model"`
	Output []oaiRespOutItem `json:"output"`
	Status string           `json:"status"`
	Usage  *oaiRespUsage    `json:"usage,omitempty"`
}

type oaiRespOutItem struct {
	Type    string             `json:"type"`
	ID      string             `json:"id,omitempty"`
	Role    string             `json:"role,omitempty"`
	Content []oaiRespOutPart   `json:"content,omitempty"`
	Summary []oaiRespSummary   `json:"summary,omitempty"`
	CallID  string             `json:"call_id,omitempty"`
	Name    string             `json:"name,omitempty"`
	Arguments string           `json:"arguments,omitempty"`
}

type oaiRespOutPart struct {
	Type string `json:"type"` // "output_text"
	Text string `json:"text,omitempty"`
}

type oaiRespSummary struct {
	Type string `json:"type"` // "summary_text"
	Text string `json:"text,omitempty"`
}

type oaiRespUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// -------- DecodeRequest --------

func (d *oaiRespDialect) DecodeRequest(body []byte) (*canonical.Request, error) {
	var req oaiRespRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("oairesponses decode request: %w", err)
	}

	canReq := &canonical.Request{
		Model:        req.Model,
		System:       req.Instructions,
		Stream:       req.Stream,
		MaxTokens:    req.MaxOutputTokens,
		Temperature:  req.Temperature,
		TopP:         req.TopP,
	}

	if req.Reasoning != nil {
		canReq.Thinking = &canonical.ThinkingConfig{
			Enabled:      true,
			BudgetTokens: effortToBudget(req.Reasoning.Effort),
		}
	}

	for _, t := range req.Tools {
		canReq.Tools = append(canReq.Tools, canonical.Tool{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
		})
	}

	// Build call_id → name map for function_call_output resolution.
	callIDToName := map[string]string{}
	for _, item := range req.Input {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		if m["type"] == "function_call" {
			callID, _ := m["call_id"].(string)
			name, _ := m["name"].(string)
			if callID != "" {
				callIDToName[callID] = name
			}
		}
	}

	// Extract system messages from input array and merge with instructions.
	var systemParts []string
	if canReq.System != "" {
		systemParts = append(systemParts, canReq.System)
	}
	for _, item := range req.Input {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := m["role"].(string)
		if role == "system" {
			if s, ok := m["content"].(string); ok && s != "" {
				systemParts = append(systemParts, s)
			}
		}
	}
	if len(systemParts) > 0 {
		canReq.System = strings.Join(systemParts, "\n")
	}

	canReq.Messages = decodeInputItems(req.Input, callIDToName)
	return canReq, nil
}

func decodeInputItems(items []interface{}, callIDToName map[string]string) []canonical.Message {
	var messages []canonical.Message

	for _, item := range items {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}

		typ, _ := m["type"].(string)
		role, _ := m["role"].(string)

		switch {
		case role == "user" || (typ == "message" && role == "user"):
			blocks := decodeInputContent(m["content"])
			if len(blocks) > 0 {
				messages = append(messages, canonical.Message{
					Role:   canonical.RoleUser,
					Blocks: blocks,
				})
			}

		case typ == "message" && role == "assistant":
			blocks := decodeOutputContent(m["content"])
			if len(blocks) > 0 {
				messages = append(messages, canonical.Message{
					Role:   canonical.RoleAssistant,
					Blocks: blocks,
				})
			}

		case typ == "reasoning":
			summaryText := extractSummaryText(m["summary"])
			if summaryText != "" {
				messages = append(messages, canonical.Message{
					Role: canonical.RoleAssistant,
					Blocks: []canonical.Block{
						canonical.ThinkingBlock{Content: summaryText, IsSummary: true},
					},
				})
			}

		case typ == "function_call":
			callID, _ := m["call_id"].(string)
			name, _ := m["name"].(string)
			argsStr, _ := m["arguments"].(string)
			var input any
			_ = json.Unmarshal([]byte(argsStr), &input)
			messages = append(messages, canonical.Message{
				Role: canonical.RoleAssistant,
				Blocks: []canonical.Block{
					canonical.ToolUseBlock{ID: callID, Name: name, Input: input},
				},
			})

		case typ == "function_call_output":
			callID, _ := m["call_id"].(string)
			output, _ := m["output"].(string)
			messages = append(messages, canonical.Message{
				Role: canonical.RoleUser,
				Blocks: []canonical.Block{
					canonical.ToolResultBlock{
						ToolUseID: callID,
						ToolName:  callIDToName[callID],
						Blocks:    []canonical.Block{canonical.TextBlock{Text: output}},
					},
				},
			})
		}
	}
	return messages
}

func decodeInputContent(raw interface{}) []canonical.Block {
	arr, ok := raw.([]interface{})
	if !ok {
		if s, ok := raw.(string); ok && s != "" {
			return []canonical.Block{canonical.TextBlock{Text: s}}
		}
		return nil
	}
	var blocks []canonical.Block
	for _, item := range arr {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		typ, _ := m["type"].(string)
		switch typ {
		case "input_text":
			text, _ := m["text"].(string)
			blocks = append(blocks, canonical.TextBlock{Text: text})
		case "input_image":
			url, _ := m["image_url"].(string)
			if url == "" {
				if img, ok := m["image_url"].(map[string]interface{}); ok {
					url, _ = img["url"].(string)
				}
			}
			if strings.HasPrefix(url, "data:") {
				mime, data := parseDataURL(url)
				blocks = append(blocks, canonical.ImageBlock{MIMEType: mime, Data: data})
			} else {
				blocks = append(blocks, canonical.ImageBlock{URL: url})
			}
		}
	}
	return blocks
}

func decodeOutputContent(raw interface{}) []canonical.Block {
	arr, ok := raw.([]interface{})
	if !ok {
		return nil
	}
	var blocks []canonical.Block
	for _, item := range arr {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		typ, _ := m["type"].(string)
		if typ == "output_text" {
			text, _ := m["text"].(string)
			blocks = append(blocks, canonical.TextBlock{Text: text})
		}
	}
	return blocks
}

func extractSummaryText(raw interface{}) string {
	arr, ok := raw.([]interface{})
	if !ok {
		return ""
	}
	var sb strings.Builder
	for _, item := range arr {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		if m["type"] == "summary_text" {
			text, _ := m["text"].(string)
			sb.WriteString(text)
		}
	}
	return sb.String()
}

func effortToBudget(effort string) int {
	switch effort {
	case "low":
		return 1000
	case "high":
		return 15000
	default:
		return 5000
	}
}

func budgetToEffort(budget int) string {
	if budget <= 0 {
		return "medium"
	}
	if budget < 2000 {
		return "low"
	}
	if budget < 8000 {
		return "medium"
	}
	return "high"
}

func parseDataURL(url string) (mimeType, data string) {
	url = strings.TrimPrefix(url, "data:")
	parts := strings.SplitN(url, ",", 2)
	if len(parts) != 2 {
		return "", url
	}
	meta := strings.TrimSuffix(parts[0], ";base64")
	return meta, parts[1]
}

// -------- EncodeRequest --------

func (d *oaiRespDialect) EncodeRequest(req *canonical.Request) ([]byte, http.Header, error) {
	var inputItems []interface{}

	for _, msg := range req.Messages {
		items := encodeMessageToInputItems(msg)
		inputItems = append(inputItems, items...)
	}

	var tools []oaiRespTool
	for _, t := range req.Tools {
		tools = append(tools, oaiRespTool{
			Type:        "function",
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
		})
	}

	out := oaiRespRequest{
		Model:        req.Model,
		Input:        inputItems,
		Instructions: req.System,
		MaxOutputTokens: req.MaxTokens,
		Temperature:  req.Temperature,
		TopP:         req.TopP,
		Stream:       req.Stream,
		Tools:        tools,
	}

	if req.Thinking != nil && req.Thinking.Enabled {
		out.Reasoning = &oaiRespReasoningCfg{
			Effort:  budgetToEffort(req.Thinking.BudgetTokens),
			Summary: "auto",
		}
	}

	body, err := json.Marshal(out)
	return body, nil, err
}

func encodeMessageToInputItems(msg canonical.Message) []interface{} {
	var items []interface{}

	switch msg.Role {
	case canonical.RoleUser:
		var toolOutputs []interface{}
		var contentParts []interface{}

		for _, blk := range msg.Blocks {
			switch b := blk.(type) {
			case canonical.ToolResultBlock:
				toolOutputs = append(toolOutputs, map[string]interface{}{
					"type":    "function_call_output",
					"call_id": b.ToolUseID,
					"output":  extractText(b.Blocks),
				})
			case canonical.TextBlock:
				contentParts = append(contentParts, map[string]interface{}{
					"type": "input_text",
					"text": b.Text,
				})
			case canonical.ImageBlock:
				url := b.URL
				if url == "" && b.Data != "" {
					url = "data:" + b.MIMEType + ";base64," + b.Data
				}
				contentParts = append(contentParts, map[string]interface{}{
					"type":      "input_image",
					"image_url": url,
				})
			}
		}

		items = append(items, toolOutputs...)
		if len(contentParts) > 0 {
			items = append(items, map[string]interface{}{
				"role":    "user",
				"content": contentParts,
			})
		}

	case canonical.RoleAssistant:
		var textParts []interface{}
		for _, blk := range msg.Blocks {
			switch b := blk.(type) {
			case canonical.ThinkingBlock:
				items = append(items, map[string]interface{}{
					"type": "reasoning",
					"summary": []interface{}{
						map[string]interface{}{"type": "summary_text", "text": b.Content},
					},
				})
			case canonical.TextBlock:
				textParts = append(textParts, map[string]interface{}{
					"type": "output_text",
					"text": b.Text,
				})
			case canonical.ToolUseBlock:
				args, _ := json.Marshal(b.Input)
				items = append(items, map[string]interface{}{
					"type":      "function_call",
					"call_id":   b.ID,
					"name":      b.Name,
					"arguments": string(args),
				})
			}
		}
		if len(textParts) > 0 {
			items = append(items, map[string]interface{}{
				"type":    "message",
				"role":    "assistant",
				"content": textParts,
			})
		}
	}

	return items
}

func extractText(blocks []canonical.Block) string {
	var sb strings.Builder
	for _, b := range blocks {
		if t, ok := b.(canonical.TextBlock); ok {
			sb.WriteString(t.Text)
		}
	}
	return sb.String()
}

// -------- BuildUpstreamPath --------

func (d *oaiRespDialect) BuildUpstreamPath(incomingPath, mappedBaseURL, model string, stream bool) string {
	return "/responses"
}

// -------- DecodeResponse --------

func (d *oaiRespDialect) DecodeResponse(body []byte) (*canonical.Response, error) {
	var resp oaiRespResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("oairesponses decode response: %w", err)
	}

	canResp := &canonical.Response{ID: resp.ID, Model: resp.Model}

	for _, item := range resp.Output {
		switch item.Type {
		case "reasoning":
			var sb strings.Builder
			for _, s := range item.Summary {
				sb.WriteString(s.Text)
			}
			if sb.Len() > 0 {
				canResp.Blocks = append(canResp.Blocks, canonical.ThinkingBlock{
					Content:   sb.String(),
					IsSummary: true,
				})
			}
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" && part.Text != "" {
					canResp.Blocks = append(canResp.Blocks, canonical.TextBlock{Text: part.Text})
				}
			}
		case "function_call":
			var input any
			_ = json.Unmarshal([]byte(item.Arguments), &input)
			canResp.Blocks = append(canResp.Blocks, canonical.ToolUseBlock{
				ID:    item.CallID,
				Name:  item.Name,
				Input: input,
			})
		}
	}

	// Determine stop reason from output types.
	hasToolUse := false
	for _, blk := range canResp.Blocks {
		if _, ok := blk.(canonical.ToolUseBlock); ok {
			hasToolUse = true
		}
	}
	if hasToolUse {
		canResp.StopReason = canonical.StopReasonToolUse
	} else {
		canResp.StopReason = canonical.StopReasonEndTurn
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

func (d *oaiRespDialect) EncodeResponse(resp *canonical.Response, clientModel string) ([]byte, error) {
	var output []oaiRespOutItem

	for _, blk := range resp.Blocks {
		switch b := blk.(type) {
		case canonical.ThinkingBlock:
			output = append(output, oaiRespOutItem{
				Type: "reasoning",
				Summary: []oaiRespSummary{
					{Type: "summary_text", Text: b.Content},
				},
			})
		case canonical.TextBlock:
			output = append(output, oaiRespOutItem{
				Type: "message",
				Role: "assistant",
				Content: []oaiRespOutPart{
					{Type: "output_text", Text: b.Text},
				},
			})
		case canonical.ToolUseBlock:
			args, _ := json.Marshal(b.Input)
			output = append(output, oaiRespOutItem{
				Type:      "function_call",
				CallID:    b.ID,
				Name:      b.Name,
				Arguments: string(args),
			})
		}
	}

	out := oaiRespResponse{
		ID:     resp.ID,
		Model:  clientModel,
		Output: output,
		Status: "completed",
	}
	if resp.Usage != nil {
		out.Usage = &oaiRespUsage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
		}
	}
	return json.Marshal(out)
}

// -------- StreamDecoder --------

type oaiRespStreamDecoder struct {
	scanner           *bufio.Scanner
	stopReason        canonical.StopReason
	usage             *canonical.TokenUsage
	currentFuncCallID string
	receivedArgsDelta bool // true if we received at least one arguments.delta event
}

func (d *oaiRespDialect) StreamDecoder(r io.Reader) translate.StreamDecoder {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)
	return &oaiRespStreamDecoder{
		scanner:    sc,
		stopReason: canonical.StopReasonEndTurn,
	}
}

func (dec *oaiRespStreamDecoder) Next() (*canonical.StreamEvent, error) {
	for {
		if !dec.scanner.Scan() {
			if err := dec.scanner.Err(); err != nil {
				return nil, err
			}
			return nil, io.EOF
		}
		line := dec.scanner.Text()

		// SSE format: "event: <name>" followed by "data: <json>"
		// We only care about data lines.
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			return &canonical.StreamEvent{
				Type:       canonical.EventDone,
				StopReason: dec.stopReason,
				Usage:      dec.usage,
			}, nil
		}

		var evt map[string]interface{}
		if err := json.Unmarshal([]byte(data), &evt); err != nil {
			continue
		}

		evtType, _ := evt["type"].(string)

		switch evtType {
		case "response.output_text.delta":
			delta, _ := evt["delta"].(string)
			if delta != "" {
				return &canonical.StreamEvent{
					Type:      canonical.EventTextDelta,
					TextDelta: delta,
				}, nil
			}

		case "response.reasoning_summary_text.delta":
			delta, _ := evt["delta"].(string)
			if delta != "" {
				return &canonical.StreamEvent{
					Type:          canonical.EventThinkingDelta,
					ThinkingDelta: delta,
				}, nil
			}

		case "response.output_item.added":
			item, _ := evt["item"].(map[string]interface{})
			if item == nil {
				continue
			}
			itemType, _ := item["type"].(string)
			if itemType == "function_call" {
				callID, _ := item["call_id"].(string)
				name, _ := item["name"].(string)
				dec.currentFuncCallID = callID
				dec.receivedArgsDelta = false
				if callID != "" || name != "" {
					return &canonical.StreamEvent{
						Type:      canonical.EventToolUseStart,
						ToolUseID: callID,
						ToolName:  name,
					}, nil
				}
			}

		case "response.function_call.arguments.delta":
			delta, _ := evt["delta"].(string)
			if delta != "" {
				dec.receivedArgsDelta = true
				return &canonical.StreamEvent{
					Type:          canonical.EventToolArgsDelta,
					ToolArgsDelta: delta,
				}, nil
			}

		case "response.function_call.arguments.done":
			// Some models send the full arguments in one shot instead of streaming deltas.
			args, _ := evt["arguments"].(string)
			if args != "" {
				return &canonical.StreamEvent{
					Type:          canonical.EventToolArgsDelta,
					ToolArgsDelta: args,
				}, nil
			}

		case "response.output_item.done":
			// Fallback: capture function_call arguments from the completed item if no deltas were sent.
			item, _ := evt["item"].(map[string]interface{})
			if item != nil {
				if itemType, _ := item["type"].(string); itemType == "function_call" {
					args, _ := item["arguments"].(string)
					if args != "" && !dec.receivedArgsDelta {
						return &canonical.StreamEvent{
							Type:          canonical.EventToolArgsDelta,
							ToolArgsDelta: args,
						}, nil
					}
				}
			}

		case "response.completed":
			response, _ := evt["response"].(map[string]interface{})
			if response != nil {
				if usage, ok := response["usage"].(map[string]interface{}); ok {
					in, _ := usage["input_tokens"].(float64)
					out, _ := usage["output_tokens"].(float64)
					dec.usage = &canonical.TokenUsage{
						InputTokens:  int(in),
						OutputTokens: int(out),
					}
				}
			}
			return &canonical.StreamEvent{
				Type:       canonical.EventDone,
				StopReason: dec.stopReason,
				Usage:      dec.usage,
			}, nil
		}
	}
}

// -------- StreamEncoder --------

type oaiRespStreamEncoder struct {
	w               http.ResponseWriter
	clientModel     string
	id              string
	flusher         http.Flusher
	msgID           string
	rsID            string
	fcID            string
	outIndex        int
	sentCreated     bool
	sentMsgStart    bool
	sentTextStart   bool
	sentThinkStart  bool
	accumulatedText string
}

func (d *oaiRespDialect) StreamEncoder(w http.ResponseWriter, clientModel string) translate.StreamEncoder {
	flusher, _ := w.(http.Flusher)
	now := time.Now().UnixNano()
	return &oaiRespStreamEncoder{
		w:           w,
		clientModel: clientModel,
		id:          fmt.Sprintf("resp_%d", now),
		msgID:       fmt.Sprintf("msg_%d", now),
		rsID:        fmt.Sprintf("rs_%d", now),
		fcID:        fmt.Sprintf("fc_%d", now),
		flusher:     flusher,
	}
}

func (enc *oaiRespStreamEncoder) ensureCreated() error {
	if enc.sentCreated {
		return nil
	}
	enc.sentCreated = true
	now := time.Now().Unix()
	respBase := map[string]any{
		"id":         enc.id,
		"object":     "response",
		"created_at": now,
		"model":      enc.clientModel,
		"status":     "in_progress",
		"output":     []any{},
	}
	if err := enc.writeSSE("response.created", map[string]any{
		"type":     "response.created",
		"response": respBase,
	}); err != nil {
		return err
	}
	return enc.writeSSE("response.in_progress", map[string]any{
		"type":     "response.in_progress",
		"response": respBase,
	})
}

func (enc *oaiRespStreamEncoder) ensureMsgStart() error {
	if err := enc.ensureCreated(); err != nil {
		return err
	}
	if enc.sentMsgStart {
		return nil
	}
	enc.sentMsgStart = true
	return enc.writeSSE("response.output_item.added", map[string]any{
		"type":         "response.output_item.added",
		"output_index": enc.outIndex,
		"item": map[string]any{
			"id":      enc.msgID,
			"type":    "message",
			"role":    "assistant",
			"status":  "in_progress",
			"content": []any{},
		},
	})
}

func (enc *oaiRespStreamEncoder) ensureTextStart() error {
	if err := enc.ensureMsgStart(); err != nil {
		return err
	}
	if enc.sentTextStart {
		return nil
	}
	enc.sentTextStart = true
	return enc.writeSSE("response.content_part.added", map[string]any{
		"type":          "response.content_part.added",
		"item_id":       enc.msgID,
		"output_index":  enc.outIndex,
		"content_index": 0,
		"part": map[string]any{
			"type": "output_text",
			"text": "",
		},
	})
}

func (enc *oaiRespStreamEncoder) Write(event *canonical.StreamEvent) error {
	switch event.Type {
	case canonical.EventTextDelta:
		if err := enc.ensureTextStart(); err != nil {
			return err
		}
		enc.accumulatedText += event.TextDelta
		return enc.writeSSE("response.output_text.delta", map[string]any{
			"type":          "response.output_text.delta",
			"item_id":       enc.msgID,
			"output_index":  enc.outIndex,
			"content_index": 0,
			"delta":         event.TextDelta,
		})

	case canonical.EventThinkingDelta:
		if err := enc.ensureCreated(); err != nil {
			return err
		}
		if !enc.sentThinkStart {
			enc.sentThinkStart = true
			if err := enc.writeSSE("response.output_item.added", map[string]any{
				"type":         "response.output_item.added",
				"output_index": enc.outIndex,
				"item": map[string]any{
					"id":      enc.rsID,
					"type":    "reasoning",
					"status":  "in_progress",
					"summary": []any{},
				},
			}); err != nil {
				return err
			}
			if err := enc.writeSSE("response.reasoning_summary_part.added", map[string]any{
				"type":          "response.reasoning_summary_part.added",
				"item_id":       enc.rsID,
				"output_index":  enc.outIndex,
				"summary_index": 0,
				"part":          map[string]any{"type": "summary_text", "text": ""},
			}); err != nil {
				return err
			}
		}
		return enc.writeSSE("response.reasoning_summary_text.delta", map[string]any{
			"type":          "response.reasoning_summary_text.delta",
			"item_id":       enc.rsID,
			"output_index":  enc.outIndex,
			"summary_index": 0,
			"delta":         event.ThinkingDelta,
		})

	case canonical.EventToolUseStart:
		if err := enc.ensureCreated(); err != nil {
			return err
		}
		// Close the message item if it was started.
		if enc.sentTextStart {
			if err := enc.closeTextPart(); err != nil {
				return err
			}
		}
		enc.outIndex++
		return enc.writeSSE("response.output_item.added", map[string]any{
			"type":         "response.output_item.added",
			"output_index": enc.outIndex,
			"item": map[string]any{
				"type":      "function_call",
				"id":        enc.fcID,
				"call_id":   event.ToolUseID,
				"name":      event.ToolName,
				"arguments": "",
				"status":    "in_progress",
			},
		})

	case canonical.EventToolArgsDelta:
		return enc.writeSSE("response.function_call.arguments.delta", map[string]any{
			"type":         "response.function_call.arguments.delta",
			"item_id":      enc.fcID,
			"output_index": enc.outIndex,
			"delta":        event.ToolArgsDelta,
		})

	case canonical.EventDone:
		if enc.sentTextStart {
			if err := enc.closeTextPart(); err != nil {
				return err
			}
		}
		if enc.sentThinkStart {
			if err := enc.writeSSE("response.reasoning_summary_text.done", map[string]any{
				"type":          "response.reasoning_summary_text.done",
				"item_id":       enc.rsID,
				"output_index":  enc.outIndex,
				"summary_index": 0,
				"text":          "",
			}); err != nil {
				return err
			}
			if err := enc.writeSSE("response.reasoning_summary_part.done", map[string]any{
				"type":          "response.reasoning_summary_part.done",
				"item_id":       enc.rsID,
				"output_index":  enc.outIndex,
				"summary_index": 0,
				"part":          map[string]any{"type": "summary_text", "text": ""},
			}); err != nil {
				return err
			}
			if err := enc.writeSSE("response.output_item.done", map[string]any{
				"type":         "response.output_item.done",
				"output_index": enc.outIndex,
				"item": map[string]any{
					"id":      enc.rsID,
					"type":    "reasoning",
					"status":  "completed",
					"summary": []any{map[string]any{"type": "summary_text", "text": ""}},
				},
			}); err != nil {
				return err
			}
		}
		var usageMap map[string]any
		if event.Usage != nil {
			usageMap = map[string]any{
				"input_tokens":  event.Usage.InputTokens,
				"output_tokens": event.Usage.OutputTokens,
				"total_tokens":  event.Usage.InputTokens + event.Usage.OutputTokens,
			}
		}
		// Build the output array for the completed response.
		var outputItems []any
		if enc.sentMsgStart {
			var contentParts []any
			if enc.sentTextStart {
				contentParts = append(contentParts, map[string]any{
					"type": "output_text",
					"text": enc.accumulatedText,
				})
			}
			outputItems = append(outputItems, map[string]any{
				"id":      enc.msgID,
				"type":    "message",
				"role":    "assistant",
				"status":  "completed",
				"content": contentParts,
			})
		}
		return enc.writeSSE("response.completed", map[string]any{
			"type": "response.completed",
			"response": map[string]any{
				"id":         enc.id,
				"object":     "response",
				"created_at": time.Now().Unix(),
				"model":      enc.clientModel,
				"status":     "completed",
				"output":     outputItems,
				"usage":      usageMap,
			},
		})
	}
	return nil
}

func (enc *oaiRespStreamEncoder) closeTextPart() error {
	if err := enc.writeSSE("response.output_text.done", map[string]any{
		"type":          "response.output_text.done",
		"item_id":       enc.msgID,
		"output_index":  enc.outIndex,
		"content_index": 0,
		"text":          enc.accumulatedText,
	}); err != nil {
		return err
	}
	if err := enc.writeSSE("response.content_part.done", map[string]any{
		"type":          "response.content_part.done",
		"item_id":       enc.msgID,
		"output_index":  enc.outIndex,
		"content_index": 0,
		"part": map[string]any{
			"type": "output_text",
			"text": enc.accumulatedText,
		},
	}); err != nil {
		return err
	}
	return enc.writeSSE("response.output_item.done", map[string]any{
		"type":         "response.output_item.done",
		"output_index": enc.outIndex,
		"item": map[string]any{
			"id":     enc.msgID,
			"type":   "message",
			"role":   "assistant",
			"status": "completed",
			"content": []any{
				map[string]any{"type": "output_text", "text": enc.accumulatedText},
			},
		},
	})
}

func (enc *oaiRespStreamEncoder) writeSSE(eventName string, data any) error {
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

func (enc *oaiRespStreamEncoder) Flush() error {
	if enc.flusher != nil {
		enc.flusher.Flush()
	}
	return nil
}
