// oye is a single-binary personal assistant that takes natural-language
// queries and dispatches them to local handlers via the embedded stitch model.
//
// One-shot:
//
//	oye weather in Madrid
//	oye Play Bad Bunny
//	oye Add milk to groceries
//
// REPL (no args):
//
//	$ oye
//	> weather in Tokyo
//	🌤  Tokyo: 18°C, light rain
//	> add bread to groceries
//	✓ added "bread" to groceries
//	> ^D
package main

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/mentasystems/stitch"
	"github.com/mentasystems/stitch/cmd/oye/handlers"
)

const banner = `oye — personal assistant powered by stitch (26M params, pure Go, offline)
type a query (e.g. "weather in Madrid", "add milk to groceries"), Ctrl-D to quit
`

func main() {
	m, err := stitch.New()
	if err != nil {
		fmt.Fprintln(os.Stderr, "init:", err)
		os.Exit(1)
	}
	dispatcher := handlers.NewDispatcher()

	if len(os.Args) > 1 {
		query := strings.Join(os.Args[1:], " ")
		runOne(m, dispatcher, query)
		return
	}
	repl(m, dispatcher)
}

func runOne(m *stitch.Model, d *handlers.Dispatcher, query string) {
	calls, err := m.Call(query, d.Tools())
	if err != nil {
		// Tolerate JSON parse errors from the model — surface the raw output
		// instead of bailing out.
		var pe *stitch.ErrParseToolCall
		if errors.As(err, &pe) {
			fmt.Fprintf(os.Stderr, "model emitted malformed JSON: %q\n", pe.RawOutput)
			return
		}
		fmt.Fprintln(os.Stderr, "call:", err)
		return
	}
	if len(calls) == 0 {
		fmt.Fprintln(os.Stderr, "no tool matched — try a different phrasing (e.g. \"weather in <place>\")")
		return
	}
	for _, c := range calls {
		if out, runErr := d.Run(c); runErr != nil {
			fmt.Fprintln(os.Stderr, "  ✗", runErr)
		} else if out != "" {
			fmt.Println(out)
		}
	}
}

func repl(m *stitch.Model, d *handlers.Dispatcher) {
	fmt.Print(banner)
	sc := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !sc.Scan() {
			fmt.Println()
			return
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if line == "exit" || line == "quit" || line == "bye" {
			return
		}
		runOne(m, d, line)
	}
}

