package model

import (
	"github.com/mentasystems/stitch/internal/tensor"
)

// Model is a loaded needle model ready for inference. It owns its scratch
// buffers, so it is NOT safe for concurrent use by multiple goroutines.
// Create one Model per goroutine that needs to run inference.
type Model struct {
	W *Weights

	// RoPE caches sized to the configured max sequence length for the encoder
	// and to the decoder's max generation length respectively.
	encRope *tensor.RoPECache
	decRope *tensor.RoPECache

	scratch *scratchBuffers

	// Cached transposed embedding (d_model, vocab) for the output projection.
	// Allocated lazily because it doubles the embedding memory footprint.
	embT []float32
}

// scratchBuffers holds reusable intermediate tensors. Sizes are upper bounds
// driven by maxEncLen and maxDecLen.
type scratchBuffers struct {
	maxEncLen int
	maxDecLen int

	// Generic per-token (d_model) workspaces.
	normedT []float32 // (max(encLen, decLen), d_model)
	attnOut []float32 // same as normedT — full attention output before residual

	// Attention intermediates sized for the worst case (encoder full-pass).
	q       []float32 // (T, d_model)
	k       []float32 // (T, kv_dim)
	v       []float32 // (T, kv_dim)
	scores  []float32 // (T,) — single-query softmax row buffer
	headOut []float32 // (T, d_model)
}

// NewModel constructs a runnable Model from loaded weights, allocating the
// scratch buffers needed to handle sequences up to maxEncLen / maxDecLen long.
func NewModel(w *Weights, maxEncLen, maxDecLen int) *Model {
	cfg := &w.Cfg
	d := cfg.DModel
	kv := cfg.KVDim()
	headDim := cfg.HeadDim()
	tMax := maxEncLen
	if maxDecLen > tMax {
		tMax = maxDecLen
	}

	m := &Model{
		W:       w,
		encRope: tensor.PrecomputeRoPE(headDim, maxEncLen, cfg.RopeTheta),
		decRope: tensor.PrecomputeRoPE(headDim, maxDecLen, cfg.RopeTheta),
		scratch: &scratchBuffers{
			maxEncLen: maxEncLen,
			maxDecLen: maxDecLen,
			normedT:   make([]float32, tMax*d),
			attnOut:   make([]float32, tMax*d),
			q:         make([]float32, tMax*d),
			k:         make([]float32, tMax*kv),
			v:         make([]float32, tMax*kv),
			scores:    make([]float32, tMax),
			headOut:   make([]float32, tMax*d),
		},
	}

	// Precompute embedding transposed for the final logits matmul:
	// embedding is (vocab, d_model); we want (d_model, vocab).
	m.embT = make([]float32, cfg.VocabSize*d)
	for r := 0; r < cfg.VocabSize; r++ {
		for c := 0; c < d; c++ {
			m.embT[c*cfg.VocabSize+r] = w.Embedding[r*d+c]
		}
	}
	return m
}
