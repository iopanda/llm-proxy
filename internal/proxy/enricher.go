package proxy

import (
	"encoding/json"
	"fmt"
	"path"
	"strings"

	"github.com/iopanda/llm-proxy/internal/config"
	"github.com/iopanda/llm-proxy/internal/translate/canonical"
)

// applySystemMods applies text replacements, static append, and optional dynamic
// progress injection to the system prompt.
func applySystemMods(system string, cfg *config.ModelConfig, msgs []canonical.Message) string {
	// 1. Text replacements.
	for old, repl := range cfg.SystemReplacements {
		system = strings.ReplaceAll(system, old, repl)
	}
	// 2. Static append.
	if cfg.SystemAppend != "" {
		if system != "" {
			system += "\n\n"
		}
		system += strings.TrimSpace(cfg.SystemAppend)
	}
	// 3. Dynamic progress context.
	if cfg.InjectProgress && len(msgs) > 0 {
		if prog := buildProgressSummary(msgs); prog != "" {
			system += "\n\n" + prog
		}
	}
	return system
}

// buildProgressSummary scans the canonical message history and produces a
// compact status block injected into the system prompt.
//
// It scans both assistant ToolUseBlocks (may be stripped by Claude Code's
// context compaction) AND user ToolResultBlocks (never stripped — always
// present), giving a more complete picture of what has been done.
func buildProgressSummary(msgs []canonical.Message) string {
	type toolOp struct {
		name   string
		cmd    string // command / file path
		result string // key outcome from tool_result (if available)
	}
	var ops []toolOp
	var cwd string

	// Index tool_use blocks by position to correlate with tool_results.
	// assistant message at index i pairs with user message at index i+1.
	type pendingOp struct {
		idx  int // message index of the assistant
		name string
		cmd  string
	}
	var pending []pendingOp

	for i, msg := range msgs {
		switch msg.Role {
		case canonical.RoleAssistant:
			for _, blk := range msg.Blocks {
				tu, ok := blk.(canonical.ToolUseBlock)
				if !ok {
					continue
				}
				args := anyToStringMap(tu.Input)
				cmd := extractCmdOrPath(tu.Name, args)
				if dir := inferCWDFromBash(cwd, cmd); tu.Name == "Bash" && dir != "" {
					cwd = dir
				}
				if path.IsAbs(cmd) && (tu.Name == "Write" || tu.Name == "Edit") {
					cwd = path.Dir(cmd)
				}
				pending = append(pending, pendingOp{idx: i, name: tu.Name, cmd: cmd})
			}

		case canonical.RoleUser:
			// Match tool_results to pending tool_use by position.
			var results []string
			for _, blk := range msg.Blocks {
				if tr, ok := blk.(canonical.ToolResultBlock); ok {
					t := strings.TrimSpace(extractBlockText(tr.Blocks))
					if t != "" {
						results = append(results, t)
					}
				}
			}
			// Flush pending ops with their results.
			for j, p := range pending {
				if p.idx == i-1 { // adjacent assistant → user pair
					result := ""
					if j < len(results) {
						result = truncateStr(results[j], 80)
					}
					ops = append(ops, toolOp{p.name, p.cmd, result})
				}
			}
			pending = nil

			// If there are tool_results without a matching pending op (tool_use was
			// stripped by compaction), still record them as "outcome" entries so the
			// model knows the results exist in history.
			if len(results) > 0 && len(pending) == 0 {
				for _, r := range results {
					if r != "" {
						ops = append(ops, toolOp{"outcome", "", truncateStr(r, 80)})
					}
				}
			}
		}
	}
	// Flush any remaining pending ops (no tool_result yet — in-flight).
	for _, p := range pending {
		ops = append(ops, toolOp{p.name, p.cmd, ""})
	}

	currentTask := lastUserText(msgs)
	if len(ops) == 0 && cwd == "" && currentTask == "" {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("---\n")

	if cwd != "" {
		sb.WriteString(fmt.Sprintf("**Current working directory**: `%s`\n\n", cwd))
	}
	if currentTask != "" {
		sb.WriteString(fmt.Sprintf("**Current task**: %s\n\n", truncateStr(currentTask, 200)))
	}

	if len(ops) > 0 {
		const maxOps = 30
		sb.WriteString("**Already completed in this session**:\n")
		start := 0
		if len(ops) > maxOps {
			start = len(ops) - maxOps
			sb.WriteString(fmt.Sprintf("_(last %d of %d operations)_\n", maxOps, len(ops)))
		}
		for _, o := range ops[start:] {
			line := fmt.Sprintf("- [%s]", o.name)
			if o.cmd != "" {
				line += " " + o.cmd
			}
			if o.result != "" {
				line += " → " + o.result
			}
			sb.WriteString(line + "\n")
		}
	}

	sb.WriteString("---")
	return sb.String()
}

// extractCmdOrPath returns the most useful short description of a tool call:
// the command for Bash, or the file path for file tools.
func extractCmdOrPath(toolName string, args map[string]any) string {
	switch toolName {
	case "Bash":
		cmd, _ := args["command"].(string)
		return truncateStr(cmd, 120)
	case "Write", "Edit", "Read":
		fp, _ := args["file_path"].(string)
		return fp
	case "Glob":
		pat, _ := args["pattern"].(string)
		return pat
	default:
		return ""
	}
}

// inferCWDFromBash extracts the most recently cd-ed directory from a Bash command string.
func inferCWDFromBash(currentCWD, cmd string) string {
	var result string
	for _, part := range splitShellChain(cmd) {
		part = strings.TrimSpace(part)
		if !strings.HasPrefix(part, "cd ") {
			continue
		}
		arg := strings.TrimSpace(part[3:])
		if f := strings.Fields(arg); len(f) > 0 {
			arg = strings.Trim(f[0], "\"'")
		}
		if arg == "" || arg == "-" || arg == ".." {
			continue
		}
		if strings.HasPrefix(arg, "/") {
			result = arg
		} else if currentCWD != "" {
			result = currentCWD + "/" + arg
		}
	}
	return result
}

// splitShellChain splits a command string on shell separators.
func splitShellChain(cmd string) []string {
	replacer := strings.NewReplacer("&&", "\n", "||", "\n", ";", "\n", "|", "\n")
	return strings.Split(replacer.Replace(cmd), "\n")
}

// lastUserText returns the text content of the last non-system-reminder user message.
func lastUserText(msgs []canonical.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		msg := msgs[i]
		if msg.Role != canonical.RoleUser {
			continue
		}
		for _, blk := range msg.Blocks {
			if tb, ok := blk.(canonical.TextBlock); ok && strings.TrimSpace(tb.Text) != "" {
				if strings.HasPrefix(tb.Text, "<system-reminder>") {
					continue
				}
				return strings.TrimSpace(tb.Text)
			}
		}
	}
	return ""
}

// isToolResultOnlyMessage returns true when a user message contains only ToolResultBlocks.
func isToolResultOnlyMessage(msg canonical.Message) bool {
	if len(msg.Blocks) == 0 {
		return false
	}
	for _, blk := range msg.Blocks {
		if _, ok := blk.(canonical.ToolResultBlock); !ok {
			return false
		}
	}
	return true
}

// truncateStr caps a string at max runes and appends "…" if truncated.
func truncateStr(s string, max int) string {
	runes := []rune(s)
	if len(runes) <= max {
		return s
	}
	return string(runes[:max]) + "…"
}

// extractBlockText concatenates the text content from a slice of canonical blocks.
func extractBlockText(blocks []canonical.Block) string {
	var sb strings.Builder
	for _, b := range blocks {
		if t, ok := b.(canonical.TextBlock); ok {
			sb.WriteString(t.Text)
		}
	}
	return sb.String()
}

// anyToStringMap converts any value to map[string]any for inspection.
func anyToStringMap(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	if v == nil {
		return nil
	}
	b, _ := json.Marshal(v)
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	return m
}
