package canonical

// Request is the dialect-neutral representation of an LLM chat request.
type Request struct {
	Model       string
	System      string
	Messages    []Message
	MaxTokens   *int
	Temperature *float64
	TopP        *float64
	Stream      bool
	Tools       []Tool
	ToolChoice  *ToolChoice
	Thinking    *ThinkingConfig
}

// Role is a message participant role.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
)

// Message is one turn in a conversation.
type Message struct {
	Role   Role
	Blocks []Block
}

// Block is a content unit within a message.
type Block interface{ isBlock() }

// TextBlock holds plain text.
type TextBlock struct{ Text string }

func (TextBlock) isBlock() {}

// ImageBlock holds an image as base64 data or a URL.
type ImageBlock struct {
	MIMEType string
	Data     string // base64-encoded
	URL      string // alternative to Data
}

func (ImageBlock) isBlock() {}

// DocumentBlock holds a document (PDF, etc.) as base64 data.
type DocumentBlock struct {
	MIMEType string
	Data     string
}

func (DocumentBlock) isBlock() {}

// ThinkingBlock holds chain-of-thought content.
// IsSummary distinguishes a condensed summary (OpenAI/Gemini) from full content (Claude).
type ThinkingBlock struct {
	Content   string
	Signature string // opaque; used by Claude for cache validation
	IsSummary bool
}

func (ThinkingBlock) isBlock() {}

// ToolUseBlock is an assistant's invocation of a tool.
type ToolUseBlock struct {
	ID    string
	Name  string
	Input any
}

func (ToolUseBlock) isBlock() {}

// ToolResultBlock carries the result of a tool invocation.
type ToolResultBlock struct {
	ToolUseID string
	ToolName  string // function name; required for Gemini; may be empty if unknown
	Blocks    []Block
	IsError   bool
}

func (ToolResultBlock) isBlock() {}

// Tool describes a callable function.
type Tool struct {
	Name        string
	Description string
	Parameters  any // JSON Schema object
}

// ToolChoice controls how the model selects tools.
// Type values: "auto", "any", "none", "tool" (specific tool via Name).
type ToolChoice struct {
	Type string
	Name string
}

// ThinkingConfig requests chain-of-thought output.
type ThinkingConfig struct {
	Enabled      bool
	BudgetTokens int
}

// Response is the dialect-neutral representation of an LLM response.
type Response struct {
	ID         string
	Model      string
	StopReason StopReason
	Blocks     []Block
	Usage      *TokenUsage
}

// StopReason explains why the model stopped generating.
type StopReason string

const (
	StopReasonEndTurn   StopReason = "end_turn"
	StopReasonMaxTokens StopReason = "max_tokens"
	StopReasonToolUse   StopReason = "tool_use"
	StopReasonStop      StopReason = "stop"
)

// TokenUsage holds token counts for a request/response cycle.
type TokenUsage struct {
	InputTokens  int
	OutputTokens int
}

// StreamEventType classifies streaming events.
type StreamEventType int

const (
	EventTextDelta StreamEventType = iota
	EventThinkingDelta
	EventToolUseStart
	EventToolArgsDelta
	EventDone
)

func (t StreamEventType) String() string {
	switch t {
	case EventTextDelta:
		return "text_delta"
	case EventThinkingDelta:
		return "thinking_delta"
	case EventToolUseStart:
		return "tool_use_start"
	case EventToolArgsDelta:
		return "tool_args_delta"
	case EventDone:
		return "done"
	default:
		return "unknown"
	}
}

// StreamEvent is a single event in a streaming response.
type StreamEvent struct {
	Type             StreamEventType
	TextDelta        string
	ThinkingDelta    string
	ToolUseID        string
	ToolName         string
	ToolArgsDelta    string
	ThoughtSignature string // Gemini: opaque signature that must accompany the function_call in history
	StopReason       StopReason
	Usage            *TokenUsage
}
