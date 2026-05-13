package model

import (
	"math"

	"github.com/mentasystems/stitch/internal/tensor"
)

const rmsEps = 1e-6

// Encode runs the 12-layer self-attention-only encoder over a batch-of-1
// sequence of token IDs. Returns the encoder output of shape (T, d_model)
// in fp32, plus the same padding mask in row-major (T_q, T_k) form for
// downstream cross-attention.
func (m *Model) Encode(tokens []int) []float32 {
	cfg := &m.W.Cfg
	T := len(tokens)
	d := cfg.DModel

	// Embedding lookup + scale by sqrt(d_model).
	scale := float32(math.Sqrt(float64(d)))
	x := make([]float32, T*d)
	for t, id := range tokens {
		emb := m.W.Embedding[id*d : (id+1)*d]
		dst := x[t*d : (t+1)*d]
		for i, v := range emb {
			dst[i] = v * scale
		}
	}

	// Padding mask: a position is valid iff tokens[t] != PAD. In golden no
	// padding ever appears, but we keep the check so this works in general.
	validKey := make([]bool, T)
	for t, id := range tokens {
		validKey[t] = id != cfg.PadTokenID
	}

	rope := m.encRope
	for li := 0; li < cfg.NumEncoderLayers; li++ {
		layer := &m.W.Encoder[li]
		// gate is the broadcast scalar applied to the attention output before
		// adding the residual.
		gate := tensor.Sigmoid(layer.AttnGate)
		m.encoderLayer(x, layer, gate, validKey, rope, T)
	}

	// Final encoder norm — applied per-token along the d_model axis.
	final := make([]float32, T*d)
	for t := 0; t < T; t++ {
		tensor.ZCRMSNorm(x[t*d:(t+1)*d], m.W.EncFinal, final[t*d:(t+1)*d], rmsEps)
	}
	return final
}

// encoderLayer applies one encoder block to x in place: pre-norm self-attention
// with a learned gate on the residual connection.
func (m *Model) encoderLayer(x []float32, layer *EncoderLayer, gate float32, validKey []bool, rope *tensor.RoPECache, T int) {
	cfg := &m.W.Cfg
	d := cfg.DModel

	// 1. Pre-norm into scratch.
	normed := m.scratch.normedT[:T*d]
	for t := 0; t < T; t++ {
		tensor.ZCRMSNorm(x[t*d:(t+1)*d], layer.PreNorm, normed[t*d:(t+1)*d], rmsEps)
	}

	// 2. Self-attention: produces (T, d_model) into m.scratch.attnOut.
	m.selfAttention(normed, &layer.Attn, validKey, rope, T, m.scratch.attnOut)

	// 3. Residual: x += gate * attnOut.
	for i := 0; i < T*d; i++ {
		x[i] += gate * m.scratch.attnOut[i]
	}
}

// selfAttention runs multi-head GQA self-attention with RoPE on input tokens
// arranged as (T, d_model). Output is written into out (also (T, d_model)).
//
// validKey is a per-position mask: positions with validKey[t]=false are
// excluded from the softmax over keys.
func (m *Model) selfAttention(input []float32, w *AttnWeights, validKey []bool, rope *tensor.RoPECache, T int, out []float32) {
	cfg := &m.W.Cfg
	d := cfg.DModel
	kv := cfg.KVDim()
	headDim := cfg.HeadDim()
	H := cfg.NumHeads
	HKV := cfg.NumKVHeads
	repeats := H / HKV

	// Project Q, K, V: (T, d_model) @ (d_model, ·) -> (T, ·)
	q := m.scratch.q[:T*d]
	k := m.scratch.k[:T*kv]
	v := m.scratch.v[:T*kv]
	tensor.MatMul(input, w.QProj, q, T, d, d)
	tensor.MatMul(input, w.KProj, k, T, d, kv)
	tensor.MatMul(input, w.VProj, v, T, d, kv)

	// Reshape (T, H, D_h) for q and (T, HKV, D_h) for k, v. We keep them as
	// (T, H, D_h)/(T, HKV, D_h) flat with stride-based access.

	// Per-head q_norm / k_norm normalises each (head_dim,) slice with the
	// shared scale across all heads of a layer.
	for t := 0; t < T; t++ {
		for h := 0; h < H; h++ {
			off := t*d + h*headDim
			tensor.ZCRMSNorm(q[off:off+headDim], w.QNorm, q[off:off+headDim], rmsEps)
		}
		for h := 0; h < HKV; h++ {
			off := t*kv + h*headDim
			tensor.ZCRMSNorm(k[off:off+headDim], w.KNorm, k[off:off+headDim], rmsEps)
		}
	}

	// Apply RoPE to Q and K (positionally).
	for t := 0; t < T; t++ {
		for h := 0; h < H; h++ {
			off := t*d + h*headDim
			tensor.ApplyRoPEInPlace(q[off:off+headDim], rope.Cos, rope.Sin, t, headDim)
		}
		for h := 0; h < HKV; h++ {
			off := t*kv + h*headDim
			tensor.ApplyRoPEInPlace(k[off:off+headDim], rope.Cos, rope.Sin, t, headDim)
		}
	}

	// Compute attention: for each query head h and query position t,
	// attn_weights[t2] = q[t,h,:] · k[t2, h_kv,:] / sqrt(head_dim), softmax,
	// then sum_t2 attn_weights[t2] * v[t2, h_kv,:].
	invScale := float32(1.0 / math.Sqrt(float64(headDim)))
	scores := m.scratch.scores[:T]
	headOut := m.scratch.headOut[:T*d] // (T, H, D_h) flat
	for h := 0; h < H; h++ {
		hKV := h / repeats // map query head to its kv head
		for t := 0; t < T; t++ {
			// Compute raw scores.
			qVec := q[t*d+h*headDim : t*d+h*headDim+headDim]
			for t2 := 0; t2 < T; t2++ {
				if !validKey[t2] {
					scores[t2] = float32(math.Inf(-1))
					continue
				}
				kVec := k[t2*kv+hKV*headDim : t2*kv+hKV*headDim+headDim]
				var dot float32
				for i := 0; i < headDim; i++ {
					dot += qVec[i] * kVec[i]
				}
				scores[t2] = dot * invScale
			}
			tensor.SoftmaxRow(scores)

			// Weighted sum over v.
			outVec := headOut[t*d+h*headDim : t*d+h*headDim+headDim]
			for i := range outVec {
				outVec[i] = 0
			}
			for t2 := 0; t2 < T; t2++ {
				weight := scores[t2]
				if weight == 0 {
					continue
				}
				vVec := v[t2*kv+hKV*headDim : t2*kv+hKV*headDim+headDim]
				for i := 0; i < headDim; i++ {
					outVec[i] += weight * vVec[i]
				}
			}
		}
	}

	// Output projection: (T, d) @ (d, d) -> (T, d).
	tensor.MatMul(headOut, w.OProj, out, T, d, d)
}
