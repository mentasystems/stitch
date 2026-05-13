# stitch

**Ship intelligence as a library: tool-calling AI in pure Go, weights embedded, no runtime required.**

```go
import "github.com/mentasystems/stitch"

m, _ := stitch.New()
calls, _ := m.Call("What's the weather in Madrid?", []stitch.Tool{{
    Name:       "get_weather",
    Parameters: map[string]string{"location": "string"},
}})

// calls[0] = {Name: "get_weather", Arguments: {"location": "Madrid"}}
```

A 26M-parameter encoder-decoder transformer that turns natural language into
function calls. Bundled in a single Go module — no Python, no C, no cgo, no
download step. Just `go get` and run.

## What you get

| | |
|---|---|
| **Single import** | `stitch.New()` — that's it |
| **Embedded model** | 50 MiB of fp16 weights baked into the binary via `go:embed` |
| **Pure Go** | Zero cgo, no system libraries, cross-compiles anywhere Go runs |
| **Fast** | ~300 ms per inference on Apple Silicon CPU (KV-cached decoder) |
| **Small** | ~26M parameters, ~60 MB binary including weights and tokenizer |
| **Greedy decoding** | Deterministic output for a given (query, tools) pair |
| **License** | MIT |

## Install

As a library:

```bash
go get github.com/mentasystems/stitch
```

As a CLI to try it without writing code:

```bash
go install github.com/mentasystems/stitch/cmd/stitch@latest
stitch "What's the weather in Madrid?"
```

Either way, the first build pulls in the 50 MiB embedded weights as part of the
module.

## API

### Tool calls

```go
type Tool struct {
    Name        string
    Parameters  map[string]string // param name -> JSON Schema type ("string", "number", ...)
    Description string            // optional, often ignored by such a small model
}

type ToolCall struct {
    Name      string
    Arguments map[string]any
}

func New() (*Model, error)
func (*Model) Call(query string, tools []Tool) ([]ToolCall, error)
func (*Model) CallRaw(query string, tools []Tool) (string, error)
```

`Call` JSON-parses the model output into `ToolCall` structs. If the model
emits malformed JSON, an `*ErrParseToolCall` is returned exposing the raw text
so the caller can decide what to do — small models occasionally fumble.

### Concurrency

A `Model` owns mutable scratch buffers and is **not** safe for concurrent
use. Either create one `Model` per goroutine, or guard access with a mutex.
`New()` is cheap to call multiple times — the embedded weights are shared
across processes via the OS page cache.

## Performance

Benchmarked on a M1 MacBook Air (single core):

| | per call |
|---|---|
| `Encode` (28 input tokens) | ~25 ms |
| `Step` (one decoded token, all 8 layers) | ~15 ms |
| Full `Call` for a 20-token output | ~330 ms |

For comparison, the JAX reference takes ~3 s on the same machine because it
re-decodes the entire `max_gen_len` buffer at every step (no KV cache). The
stitch decoder is O(L) per step, so longer outputs scale linearly.

## How it works

The model is "needle", a 26M-parameter Simple Attention Network distilled from
Gemini 3.1 by Cactus Compute. Architecture:

- BPE tokenizer (8192 vocab) with byte-fallback
- Encoder: 12 layers, GQA (8 heads, 4 KV heads), head_dim 64, RoPE, no FFN
- Decoder: 8 layers, self-attn (causal + RoPE) + cross-attn (no RoPE), no FFN
- ZCRMSNorm everywhere, tied embeddings for output projection
- d_model = 512, max sequence = 1024

stitch loads the original JAX/Flax weights from a flat binary format
(`STITCH01`), dequantises them from fp16 to fp32 once at startup, and runs
inference with a KV-cached greedy decoder.

## What it does (and doesn't)

stitch wraps a 26M-parameter model trained specifically for **single-shot
function-call extraction**. It is not a chatbot, not a code generator, and not
a reasoning engine. Concretely:

**Works well:**

- Picking one tool from a small catalogue (≤5 distinct tools)
- Extracting named entities into typed parameters (locations, contacts, song
  names, list items)
- Short, imperative phrasings ("weather in Tokyo", "Play Despacito",
  "Send Bob hello")
- Multiple languages — needle was trained multilingual; English and Spanish
  are noticeably reliable

**Does NOT work:**

- Free-form conversation — no chat training, decoder is forced to emit a
  tool-call JSON
- Code, SQL, or shell command synthesis — the model copies query fragments
  into string params, it doesn't compose syntax
- Multi-step planning ("first do X then Y") — one call per query
- Long inputs — truncated at 1024 tokens
- Phrasing variations — quality is sensitive to wording. *"weather in Tokyo"*
  works; *"What is the weather in Tokyo?"* often returns `[]`. Treat the model
  as an entity extractor for ONE specific phrasing pattern per tool

**Other quirks:**

- Numeric arguments are unreliable; the model often drops them.
- More than ~5 tools degrades selection accuracy noticeably.
- Greedy decoding only — output is deterministic for a given (query, tools).

## Re-exporting the weights

If you need to update the embedded model or change the architecture, the
exporters live in `internal/export/`:

```bash
# From the needle repo's virtualenv:
python internal/export/export.py    --checkpoint /path/to/needle.pkl --out internal/weights/weights.bin
python internal/export/tokenizer.py --model /path/to/needle.model  --out internal/weights/tokenizer.bin
```

Then `go test ./...` to verify everything still matches the golden file.

## Credits

- The model architecture and training are by [Cactus Compute](https://cactuscompute.com/),
  published as [needle](https://github.com/cactus-compute/needle). stitch is
  an independent re-implementation of inference in Go.
- See `internal/testdata/golden.json` for the reference output used to
  validate numerical parity.

## License

MIT. See `LICENSE`.
