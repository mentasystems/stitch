package model

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func loadInt8Model(t *testing.T) *Model {
	t.Helper()
	f, err := os.Open("../weights/weights_int8.bin")
	if err != nil {
		t.Skipf("weights_int8.bin not present (run internal/export/quantize.py): %v", err)
	}
	defer f.Close()
	w, err := Load(f)
	if err != nil {
		t.Fatalf("Load int8: %v", err)
	}
	return NewModel(w, 1024, 64)
}

func TestInt8Generation(t *testing.T) {
	m := loadInt8Model(t)
	data, _ := os.ReadFile("../testdata/golden.json")
	var g struct {
		EncTokens []int  `json:"enc_tokens"`
		EOSID     int    `json:"eos_id"`
		Generated []int  `json:"generated_tokens"`
		Decoded   string `json:"decoded"`
	}
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatal(err)
	}
	got := m.Generate(g.EncTokens, 64, g.EOSID, g.EOSID)
	t.Logf("int8 generated: %v", got)
	t.Logf("fp16 expected:  %v", g.Generated)
	if !reflect.DeepEqual(got, g.Generated) {
		// Show the divergence point.
		for i := 0; i < len(got) && i < len(g.Generated); i++ {
			if got[i] != g.Generated[i] {
				t.Logf("diverge at idx %d: int8=%d fp16=%d", i, got[i], g.Generated[i])
				break
			}
		}
		t.Errorf("int8 generation diverged from fp16 reference")
	}
}
