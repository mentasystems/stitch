// Package tensor provides the minimal float32 tensor ops needed by stitch's
// pure-Go inference path: fp16 dequant, matmul, ZCRMSNorm, softmax, RoPE.
package tensor

import "math"

// F16toF32 dequantises a single half-precision float to float32 (IEEE 754).
func F16toF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h&0x7C00) >> 10
	mant := uint32(h & 0x03FF)

	var bits uint32
	switch {
	case exp == 0:
		if mant == 0 {
			bits = sign
			break
		}
		// Subnormal: re-normalise.
		for mant&0x0400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &^= 0x0400
		bits = sign | ((exp + 127 - 15) << 23) | (mant << 13)
	case exp == 0x1F:
		// Inf or NaN.
		bits = sign | 0x7F800000 | (mant << 13)
	default:
		bits = sign | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(bits)
}

// DequantizeF16 reads len(out) fp16 values from raw and writes float32s to out.
// raw must be at least 2*len(out) bytes (little-endian).
func DequantizeF16(raw []byte, out []float32) {
	for i := range out {
		h := uint16(raw[2*i]) | uint16(raw[2*i+1])<<8
		out[i] = F16toF32(h)
	}
}

// MatMul computes c = a @ b for a:(M,K), b:(K,N), c:(M,N).
// All matrices are stored row-major. c is overwritten.
func MatMul(a, b, c []float32, m, k, n int) {
	// Cache-friendly i-k-j ordering: walk b row-by-row instead of column-by-column.
	for i := range c[:m*n] {
		c[i] = 0
	}
	for i := 0; i < m; i++ {
		aRow := a[i*k : (i+1)*k]
		cRow := c[i*n : (i+1)*n]
		for p := 0; p < k; p++ {
			aip := aRow[p]
			if aip == 0 {
				continue
			}
			bRow := b[p*n : (p+1)*n]
			for j := 0; j < n; j++ {
				cRow[j] += aip * bRow[j]
			}
		}
	}
}

// MatVec computes y = x @ W for x:(K,), W:(K,N), y:(N,). y is overwritten.
func MatVec(x, w, y []float32, k, n int) {
	for j := range y[:n] {
		y[j] = 0
	}
	for p := 0; p < k; p++ {
		xp := x[p]
		if xp == 0 {
			continue
		}
		wRow := w[p*n : (p+1)*n]
		for j := 0; j < n; j++ {
			y[j] += xp * wRow[j]
		}
	}
}

// ZCRMSNorm applies zero-centred RMSNorm in place: y = (1 + scale) * x / RMS(x),
// where RMS(x) = sqrt(mean(x^2) + eps). x and y must have the same length, and
// scale must be len(x).
func ZCRMSNorm(x, scale, out []float32, eps float32) {
	var sumsq float32
	for _, v := range x {
		sumsq += v * v
	}
	rms := float32(math.Sqrt(float64(sumsq/float32(len(x)) + eps)))
	invRMS := 1.0 / rms
	for i, v := range x {
		out[i] = (1 + scale[i]) * v * invRMS
	}
}

// SoftmaxRow applies numerically-stable softmax in place over a single row.
func SoftmaxRow(row []float32) {
	max := row[0]
	for _, v := range row[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range row {
		e := float32(math.Exp(float64(v - max)))
		row[i] = e
		sum += e
	}
	inv := 1.0 / sum
	for i := range row {
		row[i] *= inv
	}
}

// Sigmoid returns 1 / (1 + exp(-x)).
func Sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// RoPECache holds precomputed cos/sin tables for RoPE.
//
// shape: (seqLen, halfDim) flat row-major.
type RoPECache struct {
	Cos, Sin []float32
	SeqLen   int
	HalfDim  int
}

// PrecomputeRoPE builds cos/sin tables for a given head dim, max sequence
// length, and rotary base.
func PrecomputeRoPE(headDim, seqLen int, theta float32) *RoPECache {
	half := headDim / 2
	cache := &RoPECache{
		Cos:     make([]float32, seqLen*half),
		Sin:     make([]float32, seqLen*half),
		SeqLen:  seqLen,
		HalfDim: half,
	}
	for i := 0; i < half; i++ {
		freq := float32(math.Pow(float64(theta), -float64(2*i)/float64(headDim)))
		for t := 0; t < seqLen; t++ {
			angle := float64(float32(t) * freq)
			cache.Cos[t*half+i] = float32(math.Cos(angle))
			cache.Sin[t*half+i] = float32(math.Sin(angle))
		}
	}
	return cache
}

// ApplyRoPEInPlace rotates the per-head channels of x using cos/sin tables.
// x is shaped (T, headDim) flat; halfDim = headDim/2; pos = starting time index.
func ApplyRoPEInPlace(x []float32, cos, sin []float32, t, headDim int) {
	half := headDim / 2
	for i := 0; i < half; i++ {
		c := cos[t*half+i]
		s := sin[t*half+i]
		x1 := x[i]
		x2 := x[half+i]
		x[i] = x1*c - x2*s
		x[half+i] = x2*c + x1*s
	}
}

// ArgMax returns the index of the largest element in v.
func ArgMax(v []float32) int {
	bestIdx := 0
	best := v[0]
	for i, x := range v[1:] {
		if x > best {
			best = x
			bestIdx = i + 1
		}
	}
	return bestIdx
}
