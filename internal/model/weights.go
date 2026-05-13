// Package model loads the stitch weight file and runs needle inference in
// pure Go.
package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"

	"github.com/mentasystems/stitch/internal/tensor"
)

// Config mirrors needle's TransformerConfig — only the fields stitch consumes.
type Config struct {
	VocabSize         int     `json:"vocab_size"`
	DModel            int     `json:"d_model"`
	NumHeads          int     `json:"num_heads"`
	NumKVHeads        int     `json:"num_kv_heads"`
	NumEncoderLayers  int     `json:"num_encoder_layers"`
	NumDecoderLayers  int     `json:"num_decoder_layers"`
	DFF               int     `json:"d_ff"`
	MaxSeqLen         int     `json:"max_seq_len"`
	PadTokenID        int     `json:"pad_token_id"`
	RopeTheta         float32 `json:"rope_theta"`
	NoFeedforward     bool    `json:"no_feedforward"`
}

// HeadDim returns d_model / num_heads.
func (c *Config) HeadDim() int { return c.DModel / c.NumHeads }

// KVDim returns num_kv_heads * head_dim.
func (c *Config) KVDim() int { return c.NumKVHeads * c.HeadDim() }

// tensorInfo mirrors one entry in the JSON header of weights.bin.
type tensorInfo struct {
	Shape  []int  `json:"shape"`
	DType  string `json:"dtype"`
	Offset int    `json:"offset"`
	NBytes int    `json:"nbytes"`
}

type weightHeader struct {
	Config  json.RawMessage       `json:"config"`
	Tensors map[string]tensorInfo `json:"tensors"`
}

// AttnWeights holds one attention block's weights (already dequantised to fp32).
//
// All matrices are row-major. QProj is (d_model, d_model). KProj/VProj are
// (d_model, kv_dim). OProj is (d_model, d_model).
type AttnWeights struct {
	QProj, KProj, VProj, OProj []float32
	QNorm, KNorm               []float32 // (head_dim,)
}

// EncoderLayer holds one encoder block's weights.
type EncoderLayer struct {
	PreNorm  []float32 // (d_model,) — ZCRMSNorm scale
	AttnGate float32
	Attn     AttnWeights
}

// DecoderLayer holds one decoder block's weights.
type DecoderLayer struct {
	PreSelfNorm  []float32
	PreCrossNorm []float32
	SelfGate     float32
	CrossGate    float32
	SelfAttn     AttnWeights
	CrossAttn    AttnWeights
}

// Weights is the fully-loaded model parameters in fp32.
type Weights struct {
	Cfg Config

	Embedding []float32 // (vocab, d_model)
	EncFinal  []float32 // (d_model,)
	DecFinal  []float32 // (d_model,)

	Encoder []EncoderLayer
	Decoder []DecoderLayer
}

// numelOf returns the product of the shape's dimensions.
func numelOf(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// in1 returns the second-to-last dimension (the "in_dim" for Dense kernels).
// For (in, out) returns in. For (L, in, out) returns in.
func in1(shape []int) int {
	if len(shape) < 2 {
		return 1
	}
	return shape[len(shape)-2]
}

// Load parses a weights.bin produced by internal/export/export.py.
func Load(r io.Reader) (*Weights, error) {
	var magic [8]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic[:]) != "STITCH01" {
		return nil, fmt.Errorf("bad magic %q (want STITCH01)", magic[:])
	}
	var hlen uint32
	if err := binary.Read(r, binary.LittleEndian, &hlen); err != nil {
		return nil, fmt.Errorf("read header len: %w", err)
	}
	hdrBuf := make([]byte, hlen)
	if _, err := io.ReadFull(r, hdrBuf); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	var hdr weightHeader
	if err := json.Unmarshal(hdrBuf, &hdr); err != nil {
		return nil, fmt.Errorf("parse header: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(hdr.Config, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Compute total blob size so we know how much to read in one shot.
	var blobSize int
	for _, ti := range hdr.Tensors {
		end := ti.Offset + ti.NBytes
		if end > blobSize {
			blobSize = end
		}
	}
	blob := make([]byte, blobSize)
	if _, err := io.ReadFull(r, blob); err != nil {
		return nil, fmt.Errorf("read blob: %w", err)
	}

	// Cache name->[]float32 lookup; each tensor is independently dequantised.
	// fp16 → fp32 is a direct cast. int8 → fp32 multiplies by the matching
	// per-output-column scale tensor ("<name>_scale"). The matmul code path
	// is identical for both formats once this is done.
	tensors := make(map[string][]float32, len(hdr.Tensors))
	for name, ti := range hdr.Tensors {
		numel := 1
		for _, d := range ti.Shape {
			numel *= d
		}
		switch ti.DType {
		case "fp16":
			buf := make([]float32, numel)
			tensor.DequantizeF16(blob[ti.Offset:ti.Offset+ti.NBytes], buf)
			tensors[name] = buf
		case "int8":
			// Defer until we've parsed the matching scale tensor. We need
			// shape + scales fp16-decoded first; second pass handles them.
		default:
			return nil, fmt.Errorf("tensor %q: unsupported dtype %q", name, ti.DType)
		}
	}
	// Second pass: dequantise int8 tensors using their matching scale tensor.
	for name, ti := range hdr.Tensors {
		if ti.DType != "int8" {
			continue
		}
		scaleName := name + "_scale"
		scaleVec, ok := tensors[scaleName]
		if !ok {
			return nil, fmt.Errorf("tensor %q: missing %q", name, scaleName)
		}
		raw := blob[ti.Offset : ti.Offset+ti.NBytes]
		// Per-output-column dequant: for a tensor of shape (..., in, out), the
		// trailing 'out' axis carries the scale. Walk the int8 buffer in
		// (in, out) blocks per leading slab and multiply by scales[out].
		out := ti.Shape[len(ti.Shape)-1]
		buf := make([]float32, numelOf(ti.Shape))
		// Number of (in, out) slabs (the product of all leading dims).
		slabs := len(buf) / (in1(ti.Shape) * out)
		scalePerSlab := len(scaleVec) / slabs
		if scalePerSlab != out {
			return nil, fmt.Errorf("tensor %q: scale shape mismatch (got %d per slab, want %d)", name, scalePerSlab, out)
		}
		stride := in1(ti.Shape) * out
		for s := 0; s < slabs; s++ {
			scl := scaleVec[s*out : (s+1)*out]
			rawOff := s * stride
			dstOff := s * stride
			for i := 0; i < stride; i++ {
				val := int8(raw[rawOff+i])
				buf[dstOff+i] = float32(val) * scl[i%out]
			}
		}
		tensors[name] = buf
	}

	w := &Weights{
		Cfg:       cfg,
		Embedding: tensors["embedding/embedding"],
		EncFinal:  tensors["encoder/final_norm/scale"],
		DecFinal:  tensors["decoder/ZCRMSNorm_0/scale"],
		Encoder:   make([]EncoderLayer, cfg.NumEncoderLayers),
		Decoder:   make([]DecoderLayer, cfg.NumDecoderLayers),
	}

	dModel := cfg.DModel
	kvDim := cfg.KVDim()
	headDim := cfg.HeadDim()

	// Split each stacked layer tensor by its leading "layer" axis.
	encPreNorm := tensors["encoder/layers/EncoderBlock_0/ZCRMSNorm_0/scale"]
	encGate := tensors["encoder/layers/EncoderBlock_0/attn_gate"]
	encQ := tensors["encoder/layers/EncoderBlock_0/self_attn/q_proj/kernel"]
	encK := tensors["encoder/layers/EncoderBlock_0/self_attn/k_proj/kernel"]
	encV := tensors["encoder/layers/EncoderBlock_0/self_attn/v_proj/kernel"]
	encO := tensors["encoder/layers/EncoderBlock_0/self_attn/out_proj/kernel"]
	encQN := tensors["encoder/layers/EncoderBlock_0/self_attn/q_norm/scale"]
	encKN := tensors["encoder/layers/EncoderBlock_0/self_attn/k_norm/scale"]

	for i := 0; i < cfg.NumEncoderLayers; i++ {
		w.Encoder[i] = EncoderLayer{
			PreNorm:  encPreNorm[i*dModel : (i+1)*dModel],
			AttnGate: encGate[i],
			Attn: AttnWeights{
				QProj: encQ[i*dModel*dModel : (i+1)*dModel*dModel],
				KProj: encK[i*dModel*kvDim : (i+1)*dModel*kvDim],
				VProj: encV[i*dModel*kvDim : (i+1)*dModel*kvDim],
				OProj: encO[i*dModel*dModel : (i+1)*dModel*dModel],
				QNorm: encQN[i*headDim : (i+1)*headDim],
				KNorm: encKN[i*headDim : (i+1)*headDim],
			},
		}
	}

	decPreSelf := tensors["decoder/layers/DecoderBlock_0/ZCRMSNorm_0/scale"]
	decPreCross := tensors["decoder/layers/DecoderBlock_0/ZCRMSNorm_1/scale"]
	decSelfGate := tensors["decoder/layers/DecoderBlock_0/self_attn_gate"]
	decCrossGate := tensors["decoder/layers/DecoderBlock_0/cross_attn_gate"]

	decSaQ := tensors["decoder/layers/DecoderBlock_0/self_attn/q_proj/kernel"]
	decSaK := tensors["decoder/layers/DecoderBlock_0/self_attn/k_proj/kernel"]
	decSaV := tensors["decoder/layers/DecoderBlock_0/self_attn/v_proj/kernel"]
	decSaO := tensors["decoder/layers/DecoderBlock_0/self_attn/out_proj/kernel"]
	decSaQN := tensors["decoder/layers/DecoderBlock_0/self_attn/q_norm/scale"]
	decSaKN := tensors["decoder/layers/DecoderBlock_0/self_attn/k_norm/scale"]

	decCaQ := tensors["decoder/layers/DecoderBlock_0/cross_attn/q_proj/kernel"]
	decCaK := tensors["decoder/layers/DecoderBlock_0/cross_attn/k_proj/kernel"]
	decCaV := tensors["decoder/layers/DecoderBlock_0/cross_attn/v_proj/kernel"]
	decCaO := tensors["decoder/layers/DecoderBlock_0/cross_attn/out_proj/kernel"]
	decCaQN := tensors["decoder/layers/DecoderBlock_0/cross_attn/q_norm/scale"]
	decCaKN := tensors["decoder/layers/DecoderBlock_0/cross_attn/k_norm/scale"]

	for i := 0; i < cfg.NumDecoderLayers; i++ {
		w.Decoder[i] = DecoderLayer{
			PreSelfNorm:  decPreSelf[i*dModel : (i+1)*dModel],
			PreCrossNorm: decPreCross[i*dModel : (i+1)*dModel],
			SelfGate:     decSelfGate[i],
			CrossGate:    decCrossGate[i],
			SelfAttn: AttnWeights{
				QProj: decSaQ[i*dModel*dModel : (i+1)*dModel*dModel],
				KProj: decSaK[i*dModel*kvDim : (i+1)*dModel*kvDim],
				VProj: decSaV[i*dModel*kvDim : (i+1)*dModel*kvDim],
				OProj: decSaO[i*dModel*dModel : (i+1)*dModel*dModel],
				QNorm: decSaQN[i*headDim : (i+1)*headDim],
				KNorm: decSaKN[i*headDim : (i+1)*headDim],
			},
			CrossAttn: AttnWeights{
				QProj: decCaQ[i*dModel*dModel : (i+1)*dModel*dModel],
				KProj: decCaK[i*dModel*kvDim : (i+1)*dModel*kvDim],
				VProj: decCaV[i*dModel*kvDim : (i+1)*dModel*kvDim],
				OProj: decCaO[i*dModel*dModel : (i+1)*dModel*dModel],
				QNorm: decCaQN[i*headDim : (i+1)*headDim],
				KNorm: decCaKN[i*headDim : (i+1)*headDim],
			},
		}
	}

	// Required tensors must all exist; surface a clear error if not.
	for name, t := range map[string][]float32{
		"embedding":   w.Embedding,
		"enc_final":   w.EncFinal,
		"dec_final":   w.DecFinal,
		"enc_pre":     encPreNorm,
		"dec_self_q":  decSaQ,
		"dec_cross_q": decCaQ,
	} {
		if t == nil {
			return nil, fmt.Errorf("missing required tensor: %s", name)
		}
	}

	return w, nil
}
