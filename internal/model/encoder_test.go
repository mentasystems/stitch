package model

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func loadModel(t *testing.T) *Model {
	t.Helper()
	f, err := os.Open("../weights/weights.bin")
	if err != nil {
		t.Fatalf("open weights: %v", err)
	}
	defer f.Close()
	w, err := Load(f)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return NewModel(w, 1024, 64)
}

func TestEncoderMatchesJAX(t *testing.T) {
	m := loadModel(t)

	data, err := os.ReadFile("../testdata/encoder_out.json")
	if err != nil {
		t.Fatalf("read encoder_out.json: %v", err)
	}
	var want struct {
		EncTokens  []int     `json:"enc_tokens"`
		Shape      []int     `json:"shape"`
		EncoderOut []float32 `json:"encoder_out"`
	}
	if err := json.Unmarshal(data, &want); err != nil {
		t.Fatalf("parse: %v", err)
	}

	got := m.Encode(want.EncTokens)

	if len(got) != len(want.EncoderOut) {
		t.Fatalf("size mismatch: got %d, want %d", len(got), len(want.EncoderOut))
	}

	// Track tolerance and worst diff. JAX runs in bfloat16 (loses precision
	// every layer); we accumulate in fp32. Per-element drift up to ~0.1 is
	// expected after 12 encoder layers. What matters for greedy decoding is
	// that the *argmax* over the final logits agrees — verified separately.
	const tol = 0.15
	var maxDiff float32
	var maxIdx int
	for i, g := range got {
		d := float32(math.Abs(float64(g - want.EncoderOut[i])))
		if d > maxDiff {
			maxDiff = d
			maxIdx = i
		}
	}
	t.Logf("max abs diff = %.6f at index %d (got=%.4f want=%.4f)",
		maxDiff, maxIdx, got[maxIdx], want.EncoderOut[maxIdx])
	if maxDiff > tol {
		// Show first few diffs for diagnostics.
		shown := 0
		for i, g := range got {
			d := float32(math.Abs(float64(g - want.EncoderOut[i])))
			if d > tol && shown < 10 {
				t.Logf("  diff[%d] got=%.4f want=%.4f abs=%.4f",
					i, g, want.EncoderOut[i], d)
				shown++
			}
		}
		t.Fatalf("encoder mismatch: max diff %.4f > %.4f", maxDiff, tol)
	}
}
