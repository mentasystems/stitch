// Package handlers maps the 5 tool calls the embedded model emits onto real
// local actions: HTTP weather, background timers, file-based lists, and
// app/music launching.
package handlers

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/mentasystems/stitch"
)

// Tools returns the 5-tool catalogue the model is queried with. The schemas
// match what needle's training distribution favours: short, distinct,
// entity-extraction-shaped.
//
// global-ok: read-only catalogue, identical across goroutines.
var defaultTools = []stitch.Tool{
	{Name: "get_weather", Parameters: map[string]string{"location": "string"}},
	{Name: "set_timer", Parameters: map[string]string{"seconds": "number", "label": "string"}},
	{Name: "play_music", Parameters: map[string]string{"query": "string"}},
	{Name: "add_to_list", Parameters: map[string]string{"list": "string", "item": "string"}},
	{Name: "open_app", Parameters: map[string]string{"name": "string"}},
}

// Dispatcher routes tool calls to handler functions. Stateful: timers run in
// background goroutines, so the dispatcher must outlive the queries it serves.
type Dispatcher struct {
	listDir string
	http    *http.Client
}

// NewDispatcher returns a Dispatcher with sane defaults: ~/.oye/lists/ for
// list storage and a 5-second HTTP timeout for weather lookups.
func NewDispatcher() *Dispatcher {
	home, _ := os.UserHomeDir() // safe-ignore: empty home falls through to a relative path that still works.
	return &Dispatcher{
		listDir: filepath.Join(home, ".oye", "lists"),
		http:    &http.Client{Timeout: 5 * time.Second},
	}
}

// Tools exposes the catalogue so the caller can pass it to stitch.Call.
func (d *Dispatcher) Tools() []stitch.Tool { return defaultTools }

// Run dispatches one parsed tool call. The returned string (if non-empty) is
// printed to stdout; errors are non-fatal and printed to stderr by the caller.
//
// Needle is liberal with field names — it will happily emit "query" where the
// schema says "item" or "name". Each handler therefore looks for its canonical
// key first, then falls back to any common neighbour, then to the first string
// argument in the map.
func (d *Dispatcher) Run(c stitch.ToolCall) (string, error) {
	switch c.Name {
	case "get_weather":
		return d.weather(pickString(c.Arguments, "location", "place", "city"))
	case "set_timer":
		return d.timer(pickString(c.Arguments, "label", "name", "query"), numArg(c.Arguments, "seconds", "duration", "time"))
	case "play_music":
		return d.music(pickString(c.Arguments, "query", "song", "artist", "name"))
	case "add_to_list":
		// Model almost always emits ONE string with the item and drops the
		// list name. Be liberal: pick the item from any string, default the
		// list to a per-query-extracted list name if the model offered one
		// distinct from the item.
		item := pickString(c.Arguments, "item", "query", "value", "thing")
		list := pickStringExcluding(c.Arguments, []string{"list", "name"}, item)
		if list == "" {
			list = "default"
		}
		return d.addToList(list, item)
	case "open_app":
		return d.openApp(pickString(c.Arguments, "name", "app", "query", "item"))
	}
	return "", fmt.Errorf("unknown tool: %q", c.Name)
}

func (d *Dispatcher) weather(location string) (string, error) {
	if location == "" {
		return "", fmt.Errorf("get_weather: missing location")
	}
	endpoint := fmt.Sprintf("https://wttr.in/%s?format=3", url.PathEscape(location))
	req, err := http.NewRequest(http.MethodGet, endpoint, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "curl/8") // wttr.in returns plain text only to curl-like UAs.
	resp, err := d.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("weather lookup: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("weather: %s", strings.TrimSpace(string(body)))
	}
	return "🌤  " + strings.TrimSpace(string(body)), nil
}

func (d *Dispatcher) timer(label string, seconds int) (string, error) {
	// Needle reliably drops the seconds field; fall back to parsing a number
	// out of the label, then to a sensible default.
	if seconds <= 0 {
		seconds = guessSeconds(label)
	}
	if seconds <= 0 {
		seconds = 60
	}
	if label == "" {
		label = "timer"
	}
	dur := time.Duration(seconds) * time.Second
	go func() { // goroutine-ok: fire-and-forget timer; outliving the request is the whole point.
		time.Sleep(dur)
		notify( /* title */ "Timer: "+label, /* body */ fmt.Sprintf("%s finished", dur))
	}()
	return fmt.Sprintf("⏰ timer started: %s in %s", label, dur), nil
}

func (d *Dispatcher) music(query string) (string, error) {
	if query == "" {
		return "", fmt.Errorf("play_music: missing query")
	}
	// macOS: ask Music.app to play the search. Otherwise just print intent.
	if runtime.GOOS == "darwin" {
		script := fmt.Sprintf(`tell application "Music" to play (search playlist 1 for %q)`, query)
		_ = exec.Command("osascript", "-e", script).Run() // safe-ignore: best-effort launch; we print intent regardless.
	}
	return "▶  playing: " + query, nil
}

func (d *Dispatcher) addToList(list, item string) (string, error) {
	if list == "" {
		list = "default"
	}
	if item == "" {
		return "", fmt.Errorf("add_to_list: missing item")
	}
	if err := os.MkdirAll(d.listDir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(d.listDir, sanitizeListName(list)+".txt")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if _, writeErr := fmt.Fprintln(f, item); writeErr != nil {
		return "", writeErr
	}
	return fmt.Sprintf("✓ added %q to %s", item, list), nil
}

func (d *Dispatcher) openApp(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("open_app: missing name")
	}
	switch runtime.GOOS {
	case "darwin":
		if openErr := exec.Command("open", "-a", name).Run(); openErr != nil {
			// Don't escalate to error — printing intent is the right UX when
			// the app simply isn't installed.
			return "↗  would open: " + name + "  (not installed)", nil
		}
	case "linux":
		if openErr := exec.Command("xdg-open", strings.ToLower(name)).Run(); openErr != nil {
			return "↗  would open: " + name, nil
		}
	default:
		return "↗  would open: " + name, nil
	}
	return "↗  opening: " + name, nil
}

func notify(title, body string) {
	if runtime.GOOS != "darwin" {
		// Fall back to terminal bell + line so users still notice.
		fmt.Fprintf(os.Stderr, "\a\n[%s] %s\n", title, body)
		return
	}
	script := fmt.Sprintf(`display notification %q with title %q`, body, title)
	_ = exec.Command("osascript", "-e", script).Run() // safe-ignore: failure to notify is acceptable.
}

// pickString returns the first non-empty string value under any of keys; if
// none match, returns the first string value in args (model often invents
// field names).
func pickString(args map[string]any, keys ...string) string { // any-ok: JSON values arrive as any from the parser.
	for _, k := range keys {
		if v, ok := args[k]; ok {
			if s, isStr := v.(string); isStr && s != "" {
				return s
			}
		}
	}
	// Fallback: any string value at all.
	for _, v := range args {
		if s, ok := v.(string); ok && s != "" {
			return s
		}
	}
	return ""
}

// pickStringExcluding is like pickString but skips the given exclude value
// when scanning. Used when two slots need distinct strings (list vs item).
func pickStringExcluding(args map[string]any, keys []string, exclude string) string { // any-ok: JSON values arrive as any.
	for _, k := range keys {
		if v, ok := args[k]; ok {
			if s, isStr := v.(string); isStr && s != "" && s != exclude {
				return s
			}
		}
	}
	for _, v := range args {
		if s, ok := v.(string); ok && s != "" && s != exclude {
			return s
		}
	}
	return ""
}

// numArg returns the first numeric value found under any of keys.
func numArg(args map[string]any, keys ...string) int { // any-ok: same reason as pickString.
	for _, k := range keys {
		v, ok := args[k]
		if !ok {
			continue
		}
		switch x := v.(type) {
		case float64:
			return int(x)
		case string:
			n, err := strconv.Atoi(strings.TrimSpace(x))
			if err == nil {
				return n
			}
		}
	}
	return 0
}

// guessSeconds tries to recover a duration from a label like "25 minutes",
// "5 min", "30s", "1 hour", or "10min" (no space). Needle frequently drops the
// numeric "seconds" field and leaves the number inside the label string.
//
// global-ok: stateless regex pattern compiled once at package init.
var durationRE = regexp.MustCompile(`(\d+)\s*([a-z]*)`)

func guessSeconds(label string) int {
	for _, m := range durationRE.FindAllStringSubmatch(strings.ToLower(label), -1) {
		n, err := strconv.Atoi(m[1])
		if err != nil {
			continue
		}
		unit := m[2]
		switch {
		case strings.HasPrefix(unit, "h"):
			return n * 3600
		case strings.HasPrefix(unit, "m"): // "m", "min", "minutes"
			return n * 60
		case strings.HasPrefix(unit, "s"):
			return n
		case unit == "":
			// Bare number with no unit: assume minutes — the most common
			// timer phrasing ("25 timer" / "set 10 alarm").
			return n * 60
		}
	}
	return 0
}

func sanitizeListName(name string) string {
	name = strings.ToLower(strings.TrimSpace(name))
	name = strings.ReplaceAll(name, " ", "_")
	out := make([]rune, 0, len(name))
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_' || r == '-' {
			out = append(out, r)
		}
	}
	if len(out) == 0 {
		return "default"
	}
	return string(out)
}
