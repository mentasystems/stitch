package model

import (
	"math"

	"github.com/mentasystems/stitch/internal/tensor"
)

// DecoderState carries the KV cache and any other per-generation state. It is
// the runtime equivalent of needle's dec_buffer in JAX, but only stores the
// projected K/V matrices (no full token buffer) so each new step is O(seq) work
// rather than O(seq^2).
type DecoderState struct {
	m *Model

	// EncoderOut: (T_enc, d_model) flat — produced once by Model.Encode.
	encOut []float32
	tEnc   int

	// Precomputed per-layer cross-attention K, V projections from encOut.
	// Each is (T_enc, kv_dim) flat.
	crossK [][]float32
	crossV [][]float32

	// Per-layer self-attention KV cache: each (maxLen, kv_dim).
	selfK [][]float32
	selfV [][]float32

	// Step counter: how many decoder positions have been processed (i.e.
	// position to fill on next Step call).
	pos int
}

// NewDecoderState prepares decoder state for an encoded sequence.
func (m *Model) NewDecoderState(encOut []float32, tEnc int) *DecoderState {
	cfg := &m.W.Cfg
	kv := cfg.KVDim()
	headDim := cfg.HeadDim()
	HKV := cfg.NumKVHeads
	maxDec := m.scratch.maxDecLen

	st := &DecoderState{
		m:      m,
		encOut: encOut,
		tEnc:   tEnc,
		crossK: make([][]float32, cfg.NumDecoderLayers),
		crossV: make([][]float32, cfg.NumDecoderLayers),
		selfK:  make([][]float32, cfg.NumDecoderLayers),
		selfV:  make([][]float32, cfg.NumDecoderLayers),
	}
	for li := 0; li < cfg.NumDecoderLayers; li++ {
		// Self-attn cache: allocated to maxDec capacity, filled per step.
		st.selfK[li] = make([]float32, maxDec*kv)
		st.selfV[li] = make([]float32, maxDec*kv)

		// Cross-attn: project encOut through K, V once and apply k_norm.
		layer := &m.W.Decoder[li]
		ck := make([]float32, tEnc*kv)
		cv := make([]float32, tEnc*kv)
		tensor.MatMul(encOut, layer.CrossAttn.KProj, ck, tEnc, cfg.DModel, kv)
		tensor.MatMul(encOut, layer.CrossAttn.VProj, cv, tEnc, cfg.DModel, kv)
		// k_norm normalizes per-head along head_dim.
		for t := 0; t < tEnc; t++ {
			for h := 0; h < HKV; h++ {
				off := t*kv + h*headDim
				tensor.ZCRMSNorm(ck[off:off+headDim], layer.CrossAttn.KNorm, ck[off:off+headDim], rmsEps)
			}
		}
		st.crossK[li] = ck
		st.crossV[li] = cv
	}
	return st
}

// Step advances the decoder by one input token and returns the logits over the
// vocab for the *next* token prediction. The caller is responsible for picking
// the next token (e.g. via argmax) and feeding it back via another Step call.
func (st *DecoderState) Step(tokenID int) []float32 {
	m := st.m
	cfg := &m.W.Cfg
	d := cfg.DModel
	kv := cfg.KVDim()
	headDim := cfg.HeadDim()
	H := cfg.NumHeads
	HKV := cfg.NumKVHeads
	repeats := H / HKV
	pos := st.pos
	invScale := float32(1.0 / math.Sqrt(float64(headDim)))

	// Embed token + scale.
	x := m.scratch.normedT[:d]
	embScale := float32(math.Sqrt(float64(d)))
	emb := m.W.Embedding[tokenID*d : (tokenID+1)*d]
	for i, v := range emb {
		x[i] = v * embScale
	}

	q := m.scratch.q[:d]
	kCur := m.scratch.k[:kv]
	vCur := m.scratch.v[:kv]
	normed := m.scratch.attnOut[:d] // reuse as norm scratch

	for li := 0; li < cfg.NumDecoderLayers; li++ {
		layer := &m.W.Decoder[li]
		selfGate := tensor.Sigmoid(layer.SelfGate)
		crossGate := tensor.Sigmoid(layer.CrossGate)

		// ---- Pre-norm + Self-Attention ----
		tensor.ZCRMSNorm(x, layer.PreSelfNorm, normed, rmsEps)
		// Q (1, d), K/V (1, kv): use MatVec.
		tensor.MatVec(normed, layer.SelfAttn.QProj, q, d, d)
		tensor.MatVec(normed, layer.SelfAttn.KProj, kCur, d, kv)
		tensor.MatVec(normed, layer.SelfAttn.VProj, vCur, d, kv)

		// Per-head q_norm / k_norm normalize each head's (head_dim,) slice.
		for h := 0; h < H; h++ {
			off := h * headDim
			tensor.ZCRMSNorm(q[off:off+headDim], layer.SelfAttn.QNorm, q[off:off+headDim], rmsEps)
		}
		for h := 0; h < HKV; h++ {
			off := h * headDim
			tensor.ZCRMSNorm(kCur[off:off+headDim], layer.SelfAttn.KNorm, kCur[off:off+headDim], rmsEps)
		}

		// RoPE for q (this position) and k_cur (also this position).
		for h := 0; h < H; h++ {
			off := h * headDim
			tensor.ApplyRoPEInPlace(q[off:off+headDim], m.decRope.Cos, m.decRope.Sin, pos, headDim)
		}
		for h := 0; h < HKV; h++ {
			off := h * headDim
			tensor.ApplyRoPEInPlace(kCur[off:off+headDim], m.decRope.Cos, m.decRope.Sin, pos, headDim)
		}

		// Append k_cur, v_cur to cache at row=pos.
		copy(st.selfK[li][pos*kv:(pos+1)*kv], kCur)
		copy(st.selfV[li][pos*kv:(pos+1)*kv], vCur)

		// Self-attention over pos+1 keys: scores (pos+1,), per head.
		scores := m.scratch.scores[:pos+1]
		// headOut[h*headDim+i] aggregates one query head's output (d_model total).
		headOut := m.scratch.headOut[:d]
		for i := range headOut {
			headOut[i] = 0
		}
		for h := 0; h < H; h++ {
			hKV := h / repeats
			qVec := q[h*headDim : (h+1)*headDim]
			for t2 := 0; t2 <= pos; t2++ {
				kVec := st.selfK[li][t2*kv+hKV*headDim : t2*kv+hKV*headDim+headDim]
				var dot float32
				for i := 0; i < headDim; i++ {
					dot += qVec[i] * kVec[i]
				}
				scores[t2] = dot * invScale
			}
			tensor.SoftmaxRow(scores)
			outVec := headOut[h*headDim : (h+1)*headDim]
			for t2 := 0; t2 <= pos; t2++ {
				w := scores[t2]
				if w == 0 {
					continue
				}
				vVec := st.selfV[li][t2*kv+hKV*headDim : t2*kv+hKV*headDim+headDim]
				for i := 0; i < headDim; i++ {
					outVec[i] += w * vVec[i]
				}
			}
		}

		// Output projection + residual.
		attnOut := m.scratch.q[:d] // reuse q buffer (we're done with it for now)
		tensor.MatVec(headOut, layer.SelfAttn.OProj, attnOut, d, d)
		for i := 0; i < d; i++ {
			x[i] += selfGate * attnOut[i]
		}

		// ---- Pre-norm + Cross-Attention ----
		tensor.ZCRMSNorm(x, layer.PreCrossNorm, normed, rmsEps)
		// Q from x; K, V come from precomputed cross{K,V}.
		// Note: cross_attn has NO RoPE.
		tensor.MatVec(normed, layer.CrossAttn.QProj, q, d, d)
		for h := 0; h < H; h++ {
			off := h * headDim
			tensor.ZCRMSNorm(q[off:off+headDim], layer.CrossAttn.QNorm, q[off:off+headDim], rmsEps)
		}

		// Cross-attention: query (this position) attends over all encoder
		// positions [0..tEnc-1].
		crossScores := m.scratch.scores[:st.tEnc]
		for i := range headOut {
			headOut[i] = 0
		}
		ck := st.crossK[li]
		cv := st.crossV[li]
		for h := 0; h < H; h++ {
			hKV := h / repeats
			qVec := q[h*headDim : (h+1)*headDim]
			for t2 := 0; t2 < st.tEnc; t2++ {
				kVec := ck[t2*kv+hKV*headDim : t2*kv+hKV*headDim+headDim]
				var dot float32
				for i := 0; i < headDim; i++ {
					dot += qVec[i] * kVec[i]
				}
				crossScores[t2] = dot * invScale
			}
			tensor.SoftmaxRow(crossScores)
			outVec := headOut[h*headDim : (h+1)*headDim]
			for t2 := 0; t2 < st.tEnc; t2++ {
				w := crossScores[t2]
				if w == 0 {
					continue
				}
				vVec := cv[t2*kv+hKV*headDim : t2*kv+hKV*headDim+headDim]
				for i := 0; i < headDim; i++ {
					outVec[i] += w * vVec[i]
				}
			}
		}
		tensor.MatVec(headOut, layer.CrossAttn.OProj, attnOut, d, d)
		for i := 0; i < d; i++ {
			x[i] += crossGate * attnOut[i]
		}
	}

	// Final norm + project to vocab logits.
	finalNorm := m.scratch.normedT[d : 2*d] // separate slice from x
	tensor.ZCRMSNorm(x, m.W.DecFinal, finalNorm, rmsEps)

	logits := make([]float32, cfg.VocabSize)
	tensor.MatVec(finalNorm, m.embT, logits, d, cfg.VocabSize)

	st.pos++
	return logits
}
