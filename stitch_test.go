package stitch

import (
	"testing"
)

func TestEndToEnd(t *testing.T) {
	m, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	tools := []Tool{{
		Name:       "get_weather",
		Parameters: map[string]string{"location": "string"},
	}}
	calls, err := m.Call("What's the weather in San Francisco?", tools)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d: %+v", len(calls), calls)
	}
	if calls[0].Name != "get_weather" {
		t.Errorf("name = %q, want get_weather", calls[0].Name)
	}
	if got, ok := calls[0].Arguments["location"].(string); !ok || got != "San Francisco" {
		t.Errorf("location = %v, want 'San Francisco'", calls[0].Arguments["location"])
	}
	t.Logf("call: %+v", calls[0])
}

func TestEndToEndRaw(t *testing.T) {
	m, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	tools := []Tool{{
		Name:       "get_weather",
		Parameters: map[string]string{"location": "string"},
	}}
	raw, err := m.CallRaw("What's the weather in San Francisco?", tools)
	if err != nil {
		t.Fatalf("CallRaw: %v", err)
	}
	want := `[{"name":"get_weather","arguments":{"location":"San Francisco"}}]`
	if raw != want {
		t.Errorf("raw mismatch\n got: %q\nwant: %q", raw, want)
	}
}
