package model

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/mentasystems/stitch/internal/tensor"
)

func TestStep0MatchesGolden(t *testing.T) {
	m := loadModel(t)

	data, _ := os.ReadFile("../testdata/golden.json")
	var g struct {
		EncTokens []int       `json:"enc_tokens"`
		Step0Top5 [][2]any    `json:"step0_top5"`
		EOSID     int         `json:"eos_id"`
		Generated []int       `json:"generated_tokens"`
		Decoded   string      `json:"decoded"`
	}
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatal(err)
	}

	encOut := m.Encode(g.EncTokens)
	st := m.NewDecoderState(encOut, len(g.EncTokens))
	logits := st.Step(g.EOSID)
	got := tensor.ArgMax(logits)
	wantTop1 := int(g.Step0Top5[0][0].(float64))
	if got != wantTop1 {
		t.Errorf("step0 argmax = %d, want %d (golden top5: %v)", got, wantTop1, g.Step0Top5)
	}
}

func TestGenerateMatchesGolden(t *testing.T) {
	m := loadModel(t)

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
	if !reflect.DeepEqual(got, g.Generated) {
		t.Errorf("generated tokens mismatch\n got: %v\nwant: %v", got, g.Generated)
	}
}
