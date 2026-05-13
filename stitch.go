// Package stitch ships intelligence as a library: a 26M-parameter tool-calling
// language model in pure Go, weights embedded, no runtime required.
//
// Typical use:
//
//	m, _ := stitch.New()
//	calls, _ := m.Call("What's the weather in Madrid?", []stitch.Tool{{
//	    Name: "get_weather",
//	    Parameters: map[string]string{"location": "string"},
//	}})
//
// The first call to New takes ~50 ms to dequantise the embedded weights;
// subsequent inferences reuse the same in-memory model.
package stitch

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/mentasystems/stitch/internal/model"
	"github.com/mentasystems/stitch/internal/tokenizer"
)

//go:embed internal/weights/weights.bin internal/weights/tokenizer.bin
// global-ok: go:embed requires a package-level var; embed.FS is read-only at runtime.
var weightFS embed.FS

// Default sequence-length budgets matching needle's defaults.
const (
	defaultMaxEncLen = 1024
	defaultMaxDecLen = 512
)

// Tool describes a single function the model may call. The Parameters map
// gives parameter names and their JSON Schema type strings (e.g. "string",
// "number", "boolean").
type Tool struct {
	Name        string            `json:"name"`
	Parameters  map[string]string `json:"parameters,omitempty"`
	Description string            `json:"description,omitempty"`
}

// ToolCall is one decoded tool invocation produced by the model.
type ToolCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"` // any-ok: JSON arg values can be any type (string/number/bool/null).
}

// Model is a fully-loaded stitch model. It is NOT safe for concurrent use;
// create one Model per goroutine, or wrap access in a sync.Mutex.
type Model struct {
	mdl *model.Model
	tok *tokenizer.Tokenizer
}

// New loads the embedded needle weights and tokenizer into a runnable Model.
// The returned Model owns ~100 MB of fp32 weights.
func New() (*Model, error) {
	wBytes, err := weightFS.ReadFile("internal/weights/weights.bin")
	if err != nil {
		return nil, fmt.Errorf("stitch: read embedded weights: %w", err)
	}
	tBytes, err := weightFS.ReadFile("internal/weights/tokenizer.bin")
	if err != nil {
		return nil, fmt.Errorf("stitch: read embedded tokenizer: %w", err)
	}
	w, err := model.Load(bytes.NewReader(wBytes))
	if err != nil {
		return nil, fmt.Errorf("stitch: parse weights: %w", err)
	}
	tok, err := tokenizer.Load(bytes.NewReader(tBytes))
	if err != nil {
		return nil, fmt.Errorf("stitch: parse tokenizer: %w", err)
	}
	return &Model{
		mdl: model.NewModel(w, defaultMaxEncLen, defaultMaxDecLen),
		tok: tok,
	}, nil
}

// Call runs greedy decoding on (query, tools) and returns the parsed tool
// calls. If the model emits malformed JSON, the raw string is returned in
// ErrParseToolCall.RawOutput so the caller can decide what to do.
func (m *Model) Call(query string, tools []Tool) ([]ToolCall, error) {
	raw, err := m.CallRaw(query, tools)
	if err != nil {
		return nil, err
	}
	return parseToolCalls(raw)
}

// CallRaw runs greedy decoding and returns the raw decoded string (with the
// "<tool_call>" prefix stripped) without attempting JSON parsing.
func (m *Model) CallRaw(query string, tools []Tool) (string, error) {
	toolsJSON, err := json.Marshal(tools)
	if err != nil {
		return "", fmt.Errorf("stitch: marshal tools: %w", err)
	}
	encTokens := m.buildEncoderInput(query, /* tools */ string(toolsJSON))
	out := m.mdl.Generate(encTokens, defaultMaxDecLen, tokenizer.EOSID, tokenizer.EOSID)
	text := m.tok.Decode(out)
	// SentencePiece's add_dummy_prefix prepends a leading space.
	text = strings.TrimPrefix(text, " ")
	text = strings.TrimPrefix(text, "<tool_call>")
	text = strings.TrimSpace(text)
	return text, nil
}

// buildEncoderInput mirrors needle's _build_encoder_input:
// [query_tokens, <tools> sep, tools_tokens] truncated to maxEncLen.
func (m *Model) buildEncoderInput(query, tools string) []int {
	qToks := m.tok.Encode(query)
	tToks := m.tok.Encode(tools)
	maxQuery := defaultMaxEncLen - 2
	if len(qToks) > maxQuery {
		qToks = qToks[:maxQuery]
	}
	remaining := defaultMaxEncLen - len(qToks) - 1
	if len(tToks) > remaining {
		tToks = tToks[:remaining]
	}
	out := make([]int, 0, len(qToks)+1+len(tToks))
	out = append(out, qToks...)
	out = append(out, tokenizer.ToolsID)
	out = append(out, tToks...)
	return out
}

// ErrParseToolCall is returned by Call when the model's output is not valid
// JSON. The raw decoder text is exposed for caller inspection.
type ErrParseToolCall struct {
	RawOutput string
	Cause     error
}

func (e *ErrParseToolCall) Error() string {
	return fmt.Sprintf("stitch: tool-call JSON parse failed: %v (raw: %q)", e.Cause, e.RawOutput)
}

func (e *ErrParseToolCall) Unwrap() error { return e.Cause }

func parseToolCalls(raw string) ([]ToolCall, error) {
	if raw == "" {
		return nil, nil
	}
	var calls []ToolCall
	if err := json.Unmarshal([]byte(raw), &calls); err != nil {
		return nil, &ErrParseToolCall{RawOutput: raw, Cause: err}
	}
	return calls, nil
}
