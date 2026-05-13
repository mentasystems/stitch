// bench runs the same prompt corpus against the fp16 and int8 weight files
// and prints a side-by-side intelligence + latency comparison.
//
//	go run ./cmd/bench
//
// Outputs a markdown summary table on stdout and per-prompt diffs on stderr
// when --verbose is set.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/mentasystems/stitch/internal/bench"
	"github.com/mentasystems/stitch/internal/model"
	"github.com/mentasystems/stitch/internal/tokenizer"
)

func main() {
	var (
		fp16Path = flag.String("fp16", "internal/weights/weights.bin", "fp16 weights file")
		int8Path = flag.String("int8", "internal/weights/weights_int8.bin", "int8 weights file")
		tokPath  = flag.String("tok", "internal/weights/tokenizer.bin", "tokenizer file")
		verbose  = flag.Bool("v", false, "print per-prompt diffs to stderr")
	)
	flag.Parse()

	tok := mustTokenizer(*tokPath)
	fp16 := mustModel( /* path */ *fp16Path /* label */, "fp16")
	int8m := mustModel( /* path */ *int8Path /* label */, "int8")

	results := make([]result, 0, len(bench.Prompts))

	for _, q := range bench.Prompts {
		toks := buildEncoderInput(tok, q)

		t0 := time.Now()
		fp16Out := fp16.Generate(toks, 64, tokenizer.EOSID, tokenizer.EOSID)
		fp16Ms := float64(time.Since(t0).Microseconds()) / 1000.0

		t0 = time.Now()
		int8Out := int8m.Generate(toks, 64, tokenizer.EOSID, tokenizer.EOSID)
		int8Ms := float64(time.Since(t0).Microseconds()) / 1000.0

		results = append(results, result{
			query:      q,
			fp16Tokens: fp16Out,
			int8Tokens: int8Out,
			fp16Raw:    decodeText(tok, fp16Out),
			int8Raw:    decodeText(tok, int8Out),
			fp16Ms:     fp16Ms,
			int8Ms:     int8Ms,
		})
		if *verbose {
			same := "✓"
			if !sameTokens(fp16Out, int8Out) {
				same = "✗"
			}
			fmt.Fprintf(os.Stderr, "%s %s\n  fp16: %s\n  int8: %s\n", same, q, decodeText(tok, fp16Out), decodeText(tok, int8Out))
		}
	}

	report(results)
}

func mustModel(path, label string) *model.Model {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open %s (%s): %v\n", path, label, err)
		os.Exit(1)
	}
	defer f.Close()
	w, err := model.Load(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load %s (%s): %v\n", path, label, err)
		os.Exit(1)
	}
	return model.NewModel(w, 1024, 64)
}

func mustTokenizer(path string) *tokenizer.Tokenizer {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "tokenizer:", err)
		os.Exit(1)
	}
	defer f.Close()
	t, err := tokenizer.Load(f)
	if err != nil {
		fmt.Fprintln(os.Stderr, "tokenizer load:", err)
		os.Exit(1)
	}
	return t
}

func buildEncoderInput(t *tokenizer.Tokenizer, query string) []int {
	qToks := t.Encode(query)
	// We use empty tools here? No — the prompt set is for the bench corpus
	// with the 5-tool catalogue. Reuse the same logic stitch.go does.
	tools, _ := json.Marshal(bench.BenchTools) // safe-ignore: marshalling a static literal struct can't fail.
	tToks := t.Encode(string(tools))
	out := make([]int, 0, len(qToks)+1+len(tToks))
	out = append(out, qToks...)
	out = append(out, tokenizer.ToolsID)
	out = append(out, tToks...)
	return out
}

func decodeText(t *tokenizer.Tokenizer, ids []int) string {
	s := t.Decode(ids)
	s = strings.TrimPrefix(s, " ")
	s = strings.TrimPrefix(s, "<tool_call>")
	return strings.TrimSpace(s)
}

func sameTokens(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

type result struct {
	query      string
	fp16Tokens []int
	int8Tokens []int
	fp16Raw    string
	int8Raw    string
	fp16Ms     float64
	int8Ms     float64
}

func report(results []result) {
	n := len(results)
	if n == 0 {
		fmt.Println("no prompts")
		return
	}

	var (
		identical    int
		jsonFp16     int
		jsonInt8     int
		jsonBoth     int
		sameTool     int
		emptyFp16    int
		emptyInt8    int
		fp16Latencies []float64
		int8Latencies []float64
	)

	for _, r := range results {
		if sameTokens(r.fp16Tokens, r.int8Tokens) {
			identical++
		}
		fp16Calls, fp16OK := parseCallNames(r.fp16Raw)
		int8Calls, int8OK := parseCallNames(r.int8Raw)
		if fp16OK {
			jsonFp16++
		}
		if int8OK {
			jsonInt8++
		}
		if fp16OK && int8OK {
			jsonBoth++
			if firstName(fp16Calls) == firstName(int8Calls) {
				sameTool++
			}
		}
		if r.fp16Raw == "[]" || r.fp16Raw == "" {
			emptyFp16++
		}
		if r.int8Raw == "[]" || r.int8Raw == "" {
			emptyInt8++
		}
		fp16Latencies = append(fp16Latencies, r.fp16Ms)
		int8Latencies = append(int8Latencies, r.int8Ms)
	}

	p50f, p95f := percentile(fp16Latencies, 50), percentile(fp16Latencies, 95)
	p50i, p95i := percentile(int8Latencies, 50), percentile(int8Latencies, 95)

	fmt.Println("# stitch fp16 vs int8 benchmark")
	fmt.Println()
	fmt.Printf("Prompts: %d  (5-tool oye-style catalogue)\n", n)
	fmt.Println()
	fmt.Println("## Intelligence")
	fmt.Println()
	fmt.Println("| metric                              |     fp16 |     int8 |")
	fmt.Println("|-------------------------------------|---------:|---------:|")
	fmt.Printf("| valid JSON                          | %5d/%-2d | %5d/%-2d |\n", jsonFp16, n, jsonInt8, n)
	fmt.Printf("| empty / no tool match               | %5d/%-2d | %5d/%-2d |\n", emptyFp16, n, emptyInt8, n)
	fmt.Println()
	fmt.Printf("Token-identical fp16 ↔ int8: **%d/%d (%.0f%%)**\n", identical, n, 100*float64(identical)/float64(n))
	fmt.Printf("Same tool selected (both valid):  %d/%d (%.0f%%)\n", sameTool, jsonBoth, 100*float64(sameTool)/max(1.0, float64(jsonBoth)))
	fmt.Println()
	fmt.Println("## Latency (per prompt, ms)")
	fmt.Println()
	fmt.Println("|       |    fp16 |    int8 |")
	fmt.Println("|-------|--------:|--------:|")
	fmt.Printf("| p50   | %7.1f | %7.1f |\n", p50f, p50i)
	fmt.Printf("| p95   | %7.1f | %7.1f |\n", p95f, p95i)
	fmt.Printf("| mean  | %7.1f | %7.1f |\n", mean(fp16Latencies), mean(int8Latencies))
	fmt.Println()
	fmt.Println("(Both paths dequantise to fp32 at load and use identical matmul.")
	fmt.Println("Latency drift between them is pure measurement noise.)")
}

func parseCallNames(raw string) ([]string, bool) {
	if raw == "" || raw == "[]" {
		return nil, false
	}
	var calls []struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal([]byte(raw), &calls); err != nil {
		return nil, false
	}
	names := make([]string, len(calls))
	for i, c := range calls {
		names[i] = c.Name
	}
	return names, true
}

func firstName(names []string) string {
	if len(names) == 0 {
		return ""
	}
	return names[0]
}

func percentile(xs []float64, p int) float64 {
	if len(xs) == 0 {
		return 0
	}
	sorted := append([]float64(nil), xs...)
	sort.Float64s(sorted)
	idx := (len(sorted) - 1) * p / 100
	return sorted[idx]
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

