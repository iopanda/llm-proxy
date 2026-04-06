// Package oaichat implements the OpenAI Chat Completions API dialect.
package oaichat

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
	translate.Register(&oaiChatDialect{})
}

type oaiChatDialect struct{}

func (d *oaiChatDialect) Name() string { return "openai-chat" }

// -------- JSON types --------

type oaiRequest struct {
	Model       string       `json:"model"`
	Messages    []oaiMessage `json:"messages"`
	MaxTokens   *int         `json:"max_tokens,omitempty"`
	Temperature *float64     `json:"temperature,omitempty"`
	TopP        *float64     `json:"top_p,omitempty"`
	Stream      bool         `json:"stream,omitempty"`
	Tools       []oaiTool    `json:"tools,omitempty"`
	ToolChoice  interface{}  `json:"tool_choice,omitempty"`
}

type oaiMessage struct {
	Role       string        `json:"role"`
	Content    interface{}   `json:"content,omitempty"` // string or []oaiContentPart
	ToolCalls  []oaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
	Name       string        `json:"name,omitempty"`
}

type oaiContentPart struct {
	Type     string       `json:"type"`
	Text     string       `json:"text,omitempty"`
	ImageURL *oaiImageURL `json:"image_url,omitempty"`
}

type oaiImageURL struct {
	URL string `json:"url"`
}

type oaiToolCall struct {
	ID       string      `json:"id,omitempty"`
	Type     string      `json:"type,omitempty"`
	Function oaiFunction `json:"function"`
}

type oaiFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type oaiTool struct {
	Type     string         `json:"type"`
	Function oaiFunctionDef `json:"function"`
}

type oaiFunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type oaiResponse struct {
	ID      string      `json:"id"`
	Model   string      `json:"model"`
	Choices []oaiChoice `json:"choices"`
	Usage   *oaiUsage   `json:"usage,omitempty"`
}

type oaiChoice struct {
	Message      oaiMessage `json:"message"`
	FinishReason string     `json:"finish_reason"`
}

type oaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// -------- DecodeRequest --------

func (d *oaiChatDialect) DecodeRequest(body []byte) (*canonical.Request, error) {
	var req oaiRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("oaichat decode request: %w", err)
	}

	canReq := &canonical.Request{
		Model:       req.Model,
		Stream:      req.Stream,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		ToolChoice:  decodeToolChoice(req.ToolChoice),
	}

	for _, t := range req.Tools {
		canReq.Tools = append(canReq.Tools, canonical.Tool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		})
	}

	// Build tool call ID → name map from assistant messages for later use in tool results.
	toolCallIDToName := map[string]string{}
	for _, msg := range req.Messages {
		if msg.Role == "assistant" {
			for _, tc := range msg.ToolCalls {
				if tc.ID != "" {
					toolCallIDToName[tc.ID] = tc.Function.Name
				}
			}
		}
	}

	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			text := contentToText(msg.Content)
			if canReq.System == "" {
				canReq.System = text
			} else {
				canReq.System += "\n" + text
			}

		case "user":
			blocks := decodeContentParts(msg.Content)
			canReq.Messages = append(canReq.Messages, canonical.Message{
				Role:   canonical.RoleUser,
				Blocks: blocks,
			})

		case "assistant":
			var blocks []canonical.Block
			if msg.Content != nil {
				blocks = append(blocks, decodeContentParts(msg.Content)...)
			}
			for _, tc := range msg.ToolCalls {
				var input any
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
				blocks = append(blocks, canonical.ToolUseBlock{
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: input,
				})
			}
			canReq.Messages = append(canReq.Messages, canonical.Message{
				Role:   canonical.RoleAssistant,
				Blocks: blocks,
			})

		case "tool":
			text := contentToText(msg.Content)
			canReq.Messages = append(canReq.Messages, canonical.Message{
				Role: canonical.RoleUser,
				Blocks: []canonical.Block{
					canonical.ToolResultBlock{
						ToolUseID: msg.ToolCallID,
						ToolName:  toolCallIDToName[msg.ToolCallID],
						Blocks:    []canonical.Block{canonical.TextBlock{Text: text}},
					},
				},
			})
		}
	}

	return canReq, nil
}

func decodeContentParts(content interface{}) []canonical.Block {
	if content == nil {
		return nil
	}
	switch v := content.(type) {
	case string:
		if v == "" {
			return nil
		}
		return []canonical.Block{canonical.TextBlock{Text: v}}
	case []interface{}:
		var blocks []canonical.Block
		for _, item := range v {
			m, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			typ, _ := m["type"].(string)
			switch typ {
			case "text":
				text, _ := m["text"].(string)
				blocks = append(blocks, canonical.TextBlock{Text: text})
			case "image_url":
				imgURL, _ := m["image_url"].(map[string]interface{})
				url, _ := imgURL["url"].(string)
				if strings.HasPrefix(url, "data:") {
					mimeType, data := parseDataURL(url)
					blocks = append(blocks, canonical.ImageBlock{MIMEType: mimeType, Data: data})
				} else {
					blocks = append(blocks, canonical.ImageBlock{URL: url})
				}
			}
		}
		return blocks
	}
	return nil
}

func contentToText(content interface{}) string {
	if content == nil {
		return ""
	}
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var sb strings.Builder
		for _, item := range v {
			if m, ok := item.(map[string]interface{}); ok {
				if m["type"] == "text" {
					if text, ok := m["text"].(string); ok {
						sb.WriteString(text)
					}
				}
			}
		}
		return sb.String()
	}
	return ""
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

func decodeToolChoice(v interface{}) *canonical.ToolChoice {
	if v == nil {
		return nil
	}
	switch tc := v.(type) {
	case string:
		switch tc {
		case "required":
			return &canonical.ToolChoice{Type: "any"}
		default:
			return &canonical.ToolChoice{Type: tc}
		}
	case map[string]interface{}:
		name := ""
		if fn, ok := tc["function"].(map[string]interface{}); ok {
			name, _ = fn["name"].(string)
		}
		return &canonical.ToolChoice{Type: "tool", Name: name}
	}
	return nil
}

// -------- EncodeRequest --------

func (d *oaiChatDialect) EncodeRequest(req *canonical.Request) ([]byte, http.Header, error) {
	var messages []oaiMessage

	if req.System != "" {
		messages = append(messages, oaiMessage{Role: "system", Content: req.System})
	}

	for _, msg := range req.Messages {
		msgs, err := encodeMessage(msg)
		if err != nil {
			return nil, nil, err
		}
		messages = append(messages, msgs...)
	}

	var tools []oaiTool
	for _, t := range req.Tools {
		tools = append(tools, oaiTool{
			Type: "function",
			Function: oaiFunctionDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}

	out := oaiRequest{
		Model:       req.Model,
		Messages:    messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		Tools:       tools,
		ToolChoice:  encodeToolChoice(req.ToolChoice),
	}

	body, err := json.Marshal(out)
	return body, nil, err
}

func encodeMessage(msg canonical.Message) ([]oaiMessage, error) {
	switch msg.Role {
	case canonical.RoleUser:
		return encodeUserMessage(msg)
	case canonical.RoleAssistant:
		return encodeAssistantMessage(msg)
	case canonical.RoleSystem:
		return []oaiMessage{{Role: "system", Content: extractText(msg.Blocks)}}, nil
	}
	return nil, nil
}

func encodeUserMessage(msg canonical.Message) ([]oaiMessage, error) {
	var toolMessages []oaiMessage
	var contentParts []oaiContentPart

	for _, blk := range msg.Blocks {
		switch b := blk.(type) {
		case canonical.ToolResultBlock:
			toolMessages = append(toolMessages, oaiMessage{
				Role:       "tool",
				Content:    extractText(b.Blocks),
				ToolCallID: b.ToolUseID,
			})
		case canonical.TextBlock:
			contentParts = append(contentParts, oaiContentPart{Type: "text", Text: b.Text})
		case canonical.ImageBlock:
			url := b.URL
			if url == "" && b.Data != "" {
				url = "data:" + b.MIMEType + ";base64," + b.Data
			}
			contentParts = append(contentParts, oaiContentPart{
				Type:     "image_url",
				ImageURL: &oaiImageURL{URL: url},
			})
		}
	}

	if len(toolMessages) > 0 {
		return toolMessages, nil
	}

	var content interface{}
	if len(contentParts) == 1 && contentParts[0].Type == "text" {
		content = contentParts[0].Text
	} else if len(contentParts) > 0 {
		content = contentParts
	}

	return []oaiMessage{{Role: "user", Content: content}}, nil
}

func encodeAssistantMessage(msg canonical.Message) ([]oaiMessage, error) {
	var textContent strings.Builder
	var toolCalls []oaiToolCall

	for _, blk := range msg.Blocks {
		switch b := blk.(type) {
		case canonical.TextBlock:
			textContent.WriteString(b.Text)
		case canonical.ThinkingBlock:
			// Thinking blocks are not standard in OpenAI Chat; skip.
		case canonical.ToolUseBlock:
			args, _ := json.Marshal(b.Input)
			toolCalls = append(toolCalls, oaiToolCall{
				ID:   b.ID,
				Type: "function",
				Function: oaiFunction{
					Name:      b.Name,
					Arguments: string(args),
				},
			})
		}
	}

	m := oaiMessage{Role: "assistant"}
	if text := textContent.String(); text != "" {
		m.Content = text
	}
	if len(toolCalls) > 0 {
		m.ToolCalls = toolCalls
	}
	return []oaiMessage{m}, nil
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

func encodeToolChoice(tc *canonical.ToolChoice) interface{} {
	if tc == nil {
		return nil
	}
	switch tc.Type {
	case "auto":
		return "auto"
	case "none":
		return "none"
	case "any":
		return "required"
	case "tool":
		return map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{"name": tc.Name},
		}
	}
	return nil
}

// -------- BuildUpstreamPath --------

func (d *oaiChatDialect) BuildUpstreamPath(incomingPath, mappedBaseURL, model string, stream bool) string {
	return "/chat/completions"
}

// -------- DecodeResponse --------

func (d *oaiChatDialect) DecodeResponse(body []byte) (*canonical.Response, error) {
	var resp oaiResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("oaichat decode response: %w", err)
	}

	canResp := &canonical.Response{ID: resp.ID, Model: resp.Model}
	if len(resp.Choices) > 0 {
		ch := resp.Choices[0]
		canResp.Blocks = decodeMessageBlocks(ch.Message)
		canResp.StopReason = mapOAIStopReason(ch.FinishReason)
	}
	if resp.Usage != nil {
		canResp.Usage = &canonical.TokenUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		}
	}
	return canResp, nil
}

func decodeMessageBlocks(msg oaiMessage) []canonical.Block {
	var blocks []canonical.Block
	if msg.Content != nil {
		if text := contentToText(msg.Content); text != "" {
			blocks = append(blocks, canonical.TextBlock{Text: text})
		}
	}
	for _, tc := range msg.ToolCalls {
		var input any
		_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
		blocks = append(blocks, canonical.ToolUseBlock{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: input,
		})
	}
	return blocks
}

func mapOAIStopReason(reason string) canonical.StopReason {
	switch reason {
	case "tool_calls":
		return canonical.StopReasonToolUse
	case "length":
		return canonical.StopReasonMaxTokens
	default:
		return canonical.StopReasonStop
	}
}

func mapCanonicalToOAIStopReason(reason canonical.StopReason) string {
	switch reason {
	case canonical.StopReasonToolUse:
		return "tool_calls"
	case canonical.StopReasonMaxTokens:
		return "length"
	default:
		return "stop"
	}
}

// -------- EncodeResponse --------

func (d *oaiChatDialect) EncodeResponse(resp *canonical.Response, clientModel string) ([]byte, error) {
	var textContent strings.Builder
	var toolCalls []oaiToolCall

	for _, blk := range resp.Blocks {
		switch b := blk.(type) {
		case canonical.TextBlock:
			textContent.WriteString(b.Text)
		case canonical.ToolUseBlock:
			args, _ := json.Marshal(b.Input)
			toolCalls = append(toolCalls, oaiToolCall{
				ID:   b.ID,
				Type: "function",
				Function: oaiFunction{Name: b.Name, Arguments: string(args)},
			})
		}
	}

	msg := oaiMessage{Role: "assistant"}
	if text := textContent.String(); text != "" {
		msg.Content = text
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	var usage *oaiUsage
	if resp.Usage != nil {
		usage = &oaiUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
		}
	}

	out := oaiResponse{
		ID:    resp.ID,
		Model: clientModel,
		Choices: []oaiChoice{{
			Message:      msg,
			FinishReason: mapCanonicalToOAIStopReason(resp.StopReason),
		}},
		Usage: usage,
	}
	return json.Marshal(out)
}

// -------- StreamDecoder --------

type oaiChatStreamDecoder struct {
	scanner    *bufio.Scanner
	stopReason canonical.StopReason
	usage      *canonical.TokenUsage
}

func (d *oaiChatDialect) StreamDecoder(r io.Reader) translate.StreamDecoder {
	return &oaiChatStreamDecoder{scanner: bufio.NewScanner(r)}
}

func (dec *oaiChatStreamDecoder) Next() (*canonical.StreamEvent, error) {
	for {
		if !dec.scanner.Scan() {
			if err := dec.scanner.Err(); err != nil {
				return nil, err
			}
			return nil, io.EOF
		}
		line := dec.scanner.Text()
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

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          *string `json:"content"`
					ReasoningContent *string `json:"reasoning_content"`
					ToolCalls        []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}

		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if chunk.Usage != nil {
			dec.usage = &canonical.TokenUsage{
				InputTokens:  chunk.Usage.PromptTokens,
				OutputTokens: chunk.Usage.CompletionTokens,
			}
		}

		if len(chunk.Choices) == 0 {
			continue
		}
		ch := chunk.Choices[0]

		if ch.FinishReason != nil && *ch.FinishReason != "" {
			dec.stopReason = mapOAIStopReason(*ch.FinishReason)
		}

		if ch.Delta.Content != nil && *ch.Delta.Content != "" {
			return &canonical.StreamEvent{
				Type:      canonical.EventTextDelta,
				TextDelta: *ch.Delta.Content,
			}, nil
		}

		if ch.Delta.ReasoningContent != nil && *ch.Delta.ReasoningContent != "" {
			return &canonical.StreamEvent{
				Type:          canonical.EventThinkingDelta,
				ThinkingDelta: *ch.Delta.ReasoningContent,
			}, nil
		}

		if len(ch.Delta.ToolCalls) > 0 {
			tc := ch.Delta.ToolCalls[0]
			if tc.ID != "" {
				return &canonical.StreamEvent{
					Type:      canonical.EventToolUseStart,
					ToolUseID: tc.ID,
					ToolName:  tc.Function.Name,
				}, nil
			}
			if tc.Function.Arguments != "" {
				return &canonical.StreamEvent{
					Type:          canonical.EventToolArgsDelta,
					ToolArgsDelta: tc.Function.Arguments,
				}, nil
			}
		}
	}
}

// -------- StreamEncoder --------

type oaiChatStreamEncoder struct {
	w           http.ResponseWriter
	clientModel string
	id          string
	created     int64
	flusher     http.Flusher
	toolIndex   int
}

func (d *oaiChatDialect) StreamEncoder(w http.ResponseWriter, clientModel string) translate.StreamEncoder {
	flusher, _ := w.(http.Flusher)
	return &oaiChatStreamEncoder{
		w:           w,
		clientModel: clientModel,
		id:          fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		created:     time.Now().Unix(),
		flusher:     flusher,
		toolIndex:   -1,
	}
}

func (enc *oaiChatStreamEncoder) Write(event *canonical.StreamEvent) error {
	switch event.Type {
	case canonical.EventTextDelta:
		return enc.writeChunk(map[string]any{
			"delta":        map[string]any{"content": event.TextDelta},
			"finish_reason": nil,
		})

	case canonical.EventThinkingDelta:
		return enc.writeChunk(map[string]any{
			"delta":        map[string]any{"reasoning_content": event.ThinkingDelta},
			"finish_reason": nil,
		})

	case canonical.EventToolUseStart:
		enc.toolIndex++
		return enc.writeChunk(map[string]any{
			"delta": map[string]any{
				"tool_calls": []map[string]any{{
					"index": enc.toolIndex,
					"id":    event.ToolUseID,
					"type":  "function",
					"function": map[string]any{
						"name":      event.ToolName,
						"arguments": "",
					},
				}},
			},
			"finish_reason": nil,
		})

	case canonical.EventToolArgsDelta:
		return enc.writeChunk(map[string]any{
			"delta": map[string]any{
				"tool_calls": []map[string]any{{
					"index":    enc.toolIndex,
					"function": map[string]any{"arguments": event.ToolArgsDelta},
				}},
			},
			"finish_reason": nil,
		})

	case canonical.EventDone:
		fr := mapCanonicalToOAIStopReason(event.StopReason)
		if err := enc.writeChunk(map[string]any{
			"delta":        map[string]any{},
			"finish_reason": fr,
		}); err != nil {
			return err
		}
		_, err := fmt.Fprintf(enc.w, "data: [DONE]\n\n")
		if enc.flusher != nil {
			enc.flusher.Flush()
		}
		return err
	}
	return nil
}

func (enc *oaiChatStreamEncoder) writeChunk(choiceFields map[string]any) error {
	choiceFields["index"] = 0
	chunk := map[string]any{
		"id":      enc.id,
		"object":  "chat.completion.chunk",
		"created": enc.created,
		"model":   enc.clientModel,
		"choices": []map[string]any{choiceFields},
	}
	b, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(enc.w, "data: %s\n\n", b)
	if enc.flusher != nil {
		enc.flusher.Flush()
	}
	return err
}

func (enc *oaiChatStreamEncoder) Flush() error {
	if enc.flusher != nil {
		enc.flusher.Flush()
	}
	return nil
}
