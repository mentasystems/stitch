// Demo CLI: pass a natural-language query as the first argument, get back the
// parsed tool call on stdout.
//
//	stitch "What's the weather in Madrid?"
//	stitch Send hello to Alice
//
// Multiple positional args are joined with spaces, so quoting is optional. If
// no query is given a default sample is used so `stitch` with no args still
// produces output.
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

const defaultQuery = "What's the weather in San Francisco?"

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [query...]\n\n", os.Args[0])
		fmt.Fprintln(os.Stderr, "Runs needle (26M tool-calling model) on the given natural-language query")
		fmt.Fprintln(os.Stderr, "and prints the resulting tool call(s) as JSON on stdout.")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Example:")
		fmt.Fprintf(os.Stderr, "  %s What is the weather in Madrid?\n", os.Args[0])
	}
	flag.Parse()

	query := strings.Join(flag.Args(), " ")
	if query == "" {
		query = defaultQuery
	}

	tools := []stitch.Tool{
		{Name: "get_weather", Parameters: map[string]string{"location": "string"}},
		{Name: "send_message", Parameters: map[string]string{"to": "string", "body": "string"}},
		{Name: "set_timer", Parameters: map[string]string{"seconds": "number", "label": "string"}},
	}

	t0 := time.Now()
	m, err := stitch.New()
	if err != nil {
		fmt.Fprintln(os.Stderr, "init:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "loaded in %s\n", time.Since(t0).Round(time.Millisecond))

	t1 := time.Now()
	calls, err := m.Call(query, tools)
	if err != nil {
		fmt.Fprintln(os.Stderr, "call:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "inferred in %s\n", time.Since(t1).Round(time.Millisecond))

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if encErr := enc.Encode(calls); encErr != nil {
		fmt.Fprintln(os.Stderr, "encode:", encErr)
		os.Exit(1)
	}
}
