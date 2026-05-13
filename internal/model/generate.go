package model

import "github.com/mentasystems/stitch/internal/tensor"

// Generate runs the full encode + greedy decode loop for a single example.
// Returns the sequence of generated token IDs (excluding the start EOS and any
// trailing EOS). maxNewTokens is the cap on the number of tokens to generate.
func (m *Model) Generate(encTokens []int, maxNewTokens int, startTokenID, stopTokenID int) []int {
	encOut := m.Encode(encTokens)
	st := m.NewDecoderState(encOut, len(encTokens))

	// First step seeds the decoder with the start token (EOS for needle).
	cur := startTokenID
	out := make([]int, 0, maxNewTokens)
	for i := 0; i < maxNewTokens; i++ {
		logits := st.Step(cur)
		next := tensor.ArgMax(logits)
		if next == stopTokenID {
			break
		}
		out = append(out, next)
		cur = next
	}
	return out
}
