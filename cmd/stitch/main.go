// stitch is a demo CLI that runs the embedded needle 26M model on a query and
// prints the matched tool call(s).
//
//	stitch What's the weather in Madrid?
//	stitch Send Alice that I'm running late
//	stitch --tools-file my-tools.json My custom query goes here
//
// The default tool set covers common personal-assistant intents (weather,
// messaging, timers, music, calendar). Use --tools-file to supply your own.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/mentasystems/stitch"
)

// defaultTools is a curated set chosen to play to needle's strengths: short,
// distinct, entity-extraction-style schemas. The model excels at picking ONE
// of these from a query and extracting the relevant noun(s) — it does NOT
// generate code, sentences, or shell commands.
//
// global-ok: read-only initialiser for the default demo tool catalogue.
var defaultTools = []stitch.Tool{
	{Name: "get_weather", Parameters: map[string]string{"location": "string"}},
	{Name: "send_message", Parameters: map[string]string{"recipient": "string", "body": "string"}},
	{Name: "set_timer", Parameters: map[string]string{"seconds": "number", "label": "string"}},
	{Name: "play_music", Parameters: map[string]string{"query": "string"}},
	{Name: "create_event", Parameters: map[string]string{"title": "string", "time": "string"}},
}

const exampleHelp = `Examples (empirically verified with the default tool set):

  stitch weather in Tokyo
  stitch weather in Madrid
  stitch Send a message to Bob saying hello
  stitch Play Despacito
  stitch Put on Bad Bunny
  stitch ¿Qué tiempo hace en Sevilla?
  stitch Pon a Bad Bunny

Tips:

  - Short, imperative phrasings work better than full questions
    ("weather in Tokyo"  ✓   "What is the weather in Tokyo?"  often fails)
  - Verb-first messaging works ("Send X", "Text Y", "Play Z")
  - Numeric times/durations are unreliable — the model often drops them
  - Many sensible queries simply produce [] — that's the 26M model's ceiling,
    not a bug. Run with --raw to inspect what it generated.

Things needle does NOT do (it's a 26M tool-extraction model, not a chatbot):

  - free-form conversation
  - code or shell command generation
  - multi-step planning ("first do X then Y")
  - tools it has never seen (the demo set is fixed; use --tools-file for custom)
`

func main() {
	var (
		toolsFile = flag.String("tools-file", "", "path to a JSON file containing the tool catalogue (default: built-in personal-assistant set)")
		raw       = flag.Bool("raw", false, "print the raw model output instead of parsing it as JSON")
		verbose   = flag.Bool("v", false, "print load and inference timing on stderr")
	)
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] <query...>\n\n", os.Args[0])
		fmt.Fprintln(os.Stderr, "Runs the embedded needle model and prints the matched tool call(s) as JSON.")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Flags:")
		flag.PrintDefaults()
		fmt.Fprintln(os.Stderr)
		fmt.Fprint(os.Stderr, exampleHelp)
	}
	flag.Parse()

	query := strings.Join(flag.Args(), " ")
	if query == "" {
		flag.Usage()
		os.Exit(2)
	}

	tools := defaultTools
	if *toolsFile != "" {
		loaded, err := loadToolsFile(*toolsFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, "tools-file:", err)
			os.Exit(1)
		}
		tools = loaded
	}

	t0 := time.Now()
	m, err := stitch.New()
	if err != nil {
		fmt.Fprintln(os.Stderr, "init:", err)
		os.Exit(1)
	}
	if *verbose {
		fmt.Fprintf(os.Stderr, "loaded in %s\n", time.Since(t0).Round(time.Millisecond))
	}

	t1 := time.Now()
	if *raw {
		out, callErr := m.CallRaw(query, tools)
		if callErr != nil {
			fmt.Fprintln(os.Stderr, "call:", callErr)
			os.Exit(1)
		}
		if *verbose {
			fmt.Fprintf(os.Stderr, "inferred in %s\n", time.Since(t1).Round(time.Millisecond))
		}
		fmt.Println(out)
		return
	}

	calls, err := m.Call(query, tools)
	if err != nil {
		fmt.Fprintln(os.Stderr, "call:", err)
		os.Exit(1)
	}
	if *verbose {
		fmt.Fprintf(os.Stderr, "inferred in %s\n", time.Since(t1).Round(time.Millisecond))
	}

	if len(calls) == 0 {
		fmt.Fprintln(os.Stderr, "no tool matched — try rephrasing, or run with --raw to see the model's output")
		os.Exit(3)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if encErr := enc.Encode(calls); encErr != nil {
		fmt.Fprintln(os.Stderr, "encode:", encErr)
		os.Exit(1)
	}
}

func loadToolsFile(path string) ([]stitch.Tool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var tools []stitch.Tool
	if jsonErr := json.Unmarshal(data, &tools); jsonErr != nil {
		return nil, fmt.Errorf("parse %s: %w", path, jsonErr)
	}
	if len(tools) == 0 {
		return nil, fmt.Errorf("no tools in %s", path)
	}
	return tools, nil
}
