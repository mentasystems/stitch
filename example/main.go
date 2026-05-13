// Example: a one-shot CLI that reads a query, prints the parsed tool call.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/mentasystems/stitch"
)

func main() {
	query := flag.String("q", "What's the weather in San Francisco?", "user query")
	flag.Parse()

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
	calls, err := m.Call(*query, tools)
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
