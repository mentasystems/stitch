package tokenizer

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func loadTestTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	f, err := os.Open("../weights/tokenizer.bin")
	if err != nil {
		t.Fatalf("open tokenizer.bin: %v", err)
	}
	defer f.Close()
	tok, err := Load(f)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return tok
}

func TestEncodeGolden(t *testing.T) {
	tok := loadTestTokenizer(t)

	if got := tok.VocabSize(); got != 8192 {
		t.Errorf("VocabSize() = %d, want 8192", got)
	}

	// First 14 tokens of the golden enc_tokens are the query encoding
	// "What's the weather in San Francisco?" (before <tools> sep and tools JSON).
	wantQuery := []int{4279, 8066, 8046, 302, 1149, 362, 711, 327, 1295, 1075, 378, 275, 8047, 8105}
	got := tok.Encode("What's the weather in San Francisco?")
	if !reflect.DeepEqual(got, wantQuery) {
		t.Errorf("Encode query mismatch\n got: %v\nwant: %v", got, wantQuery)
	}

	wantTools := []int{8041, 5, 8071, 271, 294, 264, 358, 8062, 1331, 8039, 8059, 8072}
	got = tok.Encode(`<tools>[{"name":"get_weather"}]`)
	if !reflect.DeepEqual(got, wantTools) {
		t.Errorf("Encode tools mismatch\n got: %v\nwant: %v", got, wantTools)
	}

	wantHello := []int{706, 8055, 363, 5338, 745}
	got = tok.Encode("hello world")
	if !reflect.DeepEqual(got, wantHello) {
		t.Errorf("Encode hello mismatch\n got: %v\nwant: %v", got, wantHello)
	}
}

func TestEncodeFullGoldenInput(t *testing.T) {
	tok := loadTestTokenizer(t)

	// Verify the full encoder-input matches golden.json.
	data, err := os.ReadFile("../testdata/golden.json")
	if err != nil {
		t.Fatalf("read golden: %v", err)
	}
	var golden struct {
		Query      string `json:"query"`
		Tools      string `json:"tools"`
		EncTokens  []int  `json:"enc_tokens"`
		ToolsSepID int    `json:"tools_sep_id"`
	}
	if err := json.Unmarshal(data, &golden); err != nil {
		t.Fatalf("parse golden: %v", err)
	}

	queryIDs := tok.Encode(golden.Query)
	toolsIDs := tok.Encode(golden.Tools)
	full := append(append(queryIDs, golden.ToolsSepID), toolsIDs...)
	if !reflect.DeepEqual(full, golden.EncTokens) {
		t.Errorf("full encoder input mismatch\n got: %v\nwant: %v", full, golden.EncTokens)
	}
}

func TestDecodeRoundtrip(t *testing.T) {
	tok := loadTestTokenizer(t)
	cases := []string{
		"What's the weather in San Francisco?",
		"hello world",
		"Send 'hello' to Alice",
	}
	for _, in := range cases {
		ids := tok.Encode(in)
		out := tok.Decode(ids)
		// SentencePiece prepends a leading space due to add_dummy_prefix.
		if out != " "+in {
			t.Errorf("roundtrip mismatch\n  in:  %q\n  out: %q", in, out)
		}
	}
}
