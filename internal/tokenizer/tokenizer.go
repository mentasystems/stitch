// Package tokenizer is a pure-Go SentencePiece BPE tokenizer that matches the
// reference Python implementation used by needle (BPE + byte_fallback +
// user-defined symbols + identity normalization + add_dummy_prefix).
package tokenizer

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"unicode/utf8"
)

// SentencePiece type codes (mirrors our export.py).
const (
	TypeNormal      byte = 0
	TypeUnknown     byte = 1
	TypeControl     byte = 2
	TypeUserDefined byte = 3
	TypeByte        byte = 4
	TypeUnused      byte = 5
)

// SentencePiece's whitespace marker character: U+2581 (LOWER ONE EIGHTH BLOCK).
const spaceMarker = "▁"

// Special token IDs hardcoded by needle (see internal/export/tokenizer.py).
const (
	PadID      = 0
	EOSID      = 1
	BOSID      = 2
	UnkID      = 3
	ToolCallID = 4
	ToolsID    = 5
)

// Piece is one entry in the SentencePiece vocabulary.
type Piece struct {
	Str   string
	Score float32
	Type  byte
}

// Tokenizer encodes text to int IDs and back, matching SentencePiece BPE.
type Tokenizer struct {
	pieces         []Piece
	pieceID        map[string]int32 // piece string -> vocab ID
	byteToID       [256]int32       // byte fallback table: 0xNN -> ID of "<0xNN>"
	userDefined    []string         // user-defined pieces sorted by length desc
	byteFallback   bool
	addDummyPrefix bool
}

// Load parses a tokenizer.bin written by internal/export/tokenizer.py.
func Load(r io.Reader) (*Tokenizer, error) {
	var magic [8]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic[:]) != "STITCK01" {
		return nil, fmt.Errorf("bad magic %q (want STITCK01)", magic[:])
	}
	var vocabSize uint32
	if err := binary.Read(r, binary.LittleEndian, &vocabSize); err != nil {
		return nil, err
	}
	var flags [4]byte
	if _, err := io.ReadFull(r, flags[:]); err != nil {
		return nil, err
	}
	t := &Tokenizer{
		pieces:         make([]Piece, vocabSize),
		pieceID:        make(map[string]int32, vocabSize),
		byteFallback:   flags[0] != 0,
		addDummyPrefix: flags[1] != 0,
	}
	for i := range t.byteToID {
		t.byteToID[i] = -1
	}
	for i := uint32(0); i < vocabSize; i++ {
		var score float32
		if err := binary.Read(r, binary.LittleEndian, &score); err != nil {
			return nil, fmt.Errorf("piece %d score: %w", i, err)
		}
		var typeByte [1]byte
		if _, err := io.ReadFull(r, typeByte[:]); err != nil {
			return nil, fmt.Errorf("piece %d type: %w", i, err)
		}
		var n uint16
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return nil, fmt.Errorf("piece %d len: %w", i, err)
		}
		buf := make([]byte, n)
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("piece %d body: %w", i, err)
		}
		p := Piece{Str: string(buf), Score: score, Type: typeByte[0]}
		t.pieces[i] = p
		t.pieceID[p.Str] = int32(i)
		switch p.Type {
		case TypeUserDefined:
			t.userDefined = append(t.userDefined, p.Str)
		case TypeByte:
			// piece is "<0xNN>"
			if len(p.Str) == 6 && p.Str[0] == '<' && p.Str[5] == '>' {
				var b byte
				if _, err := fmt.Sscanf(p.Str, "<0x%02X>", &b); err == nil {
					t.byteToID[b] = int32(i)
				}
			}
		}
	}
	// Sort user-defined symbols by descending length to match longest-first.
	sort.Slice(t.userDefined, func(i, j int) bool {
		return len(t.userDefined[i]) > len(t.userDefined[j])
	})
	return t, nil
}

// VocabSize returns the number of pieces in the vocabulary.
func (t *Tokenizer) VocabSize() int { return len(t.pieces) }

// PieceOf returns the piece string for id (no validation).
func (t *Tokenizer) PieceOf(id int) string {
	if id < 0 || id >= len(t.pieces) {
		return ""
	}
	return t.pieces[id].Str
}

// Encode converts text to a sequence of int IDs.
func (t *Tokenizer) Encode(text string) []int {
	// Whitespace pre-processing: spaces -> ▁, then add_dummy_prefix ▁ in front.
	text = strings.ReplaceAll(text, " ", spaceMarker)
	if t.addDummyPrefix {
		text = spaceMarker + text
	}

	var ids []int
	i := 0
	for i < len(text) {
		// Check user-defined symbols (longest-first) at this position.
		matched := false
		for _, sym := range t.userDefined {
			if i+len(sym) <= len(text) && text[i:i+len(sym)] == sym {
				ids = append(ids, int(t.pieceID[sym]))
				i += len(sym)
				matched = true
				break
			}
		}
		if matched {
			continue
		}
		// Find end of segment: next user-defined match or end-of-string.
		segEnd := t.findNextUserDefined(text, i)
		segment := text[i:segEnd]
		ids = append(ids, t.bpeEncode(segment)...)
		i = segEnd
	}
	return ids
}

func (t *Tokenizer) findNextUserDefined(text string, from int) int {
	if len(t.userDefined) == 0 {
		return len(text)
	}
	for j := from + 1; j < len(text); j++ {
		for _, sym := range t.userDefined {
			if j+len(sym) <= len(text) && text[j:j+len(sym)] == sym {
				return j
			}
		}
	}
	return len(text)
}

// bpeEncode runs the SentencePiece BPE merge loop on a segment that contains
// no user-defined symbols.
func (t *Tokenizer) bpeEncode(s string) []int {
	if s == "" {
		return nil
	}
	// Split into single-codepoint pieces.
	runes := make([]string, 0, len(s))
	for _, r := range s {
		runes = append(runes, string(r))
	}

	// Greedy BPE merge: repeatedly merge the adjacent pair with the highest
	// score that forms a known vocab piece.
	for {
		bestScore := float32(math.Inf(-1))
		bestIdx := -1
		for k := 0; k < len(runes)-1; k++ {
			merged := runes[k] + runes[k+1]
			if id, ok := t.pieceID[merged]; ok {
				if score := t.pieces[id].Score; score > bestScore {
					bestScore = score
					bestIdx = k
				}
			}
		}
		if bestIdx == -1 {
			break
		}
		runes[bestIdx] = runes[bestIdx] + runes[bestIdx+1]
		runes = append(runes[:bestIdx+1], runes[bestIdx+2:]...)
	}

	// Resolve each piece to an ID, falling back to byte tokens for unknowns.
	out := make([]int, 0, len(runes))
	for _, p := range runes {
		if id, ok := t.pieceID[p]; ok {
			out = append(out, int(id))
			continue
		}
		if t.byteFallback {
			for _, b := range []byte(p) {
				id := t.byteToID[b]
				if id < 0 {
					out = append(out, UnkID)
				} else {
					out = append(out, int(id))
				}
			}
			continue
		}
		out = append(out, UnkID)
	}
	return out
}

// Decode reverses Encode: turns IDs back into a UTF-8 string. Byte tokens are
// re-assembled into UTF-8 (matching SentencePiece's behaviour), and ▁ is
// converted back to space. The leading dummy-prefix space is preserved.
func (t *Tokenizer) Decode(ids []int) string {
	var buf strings.Builder
	var pendingBytes []byte
	flush := func() {
		if len(pendingBytes) > 0 {
			buf.Write(pendingBytes)
			pendingBytes = pendingBytes[:0]
		}
	}
	for _, id := range ids {
		if id < 0 || id >= len(t.pieces) {
			continue
		}
		p := t.pieces[id]
		switch p.Type {
		case TypeByte:
			// piece is "<0xNN>" → emit raw byte
			var b byte
			if _, err := fmt.Sscanf(p.Str, "<0x%02X>", &b); err == nil {
				pendingBytes = append(pendingBytes, b)
			}
		case TypeControl:
			flush()
			// CONTROL tokens (BOS/EOS/PAD) are dropped from the decoded text.
		default:
			flush()
			buf.WriteString(p.Str)
		}
	}
	flush()
	out := buf.String()
	out = strings.ReplaceAll(out, spaceMarker, " ")
	return out
}

// ensure utf8 is referenced so it stays in the build even if we later strip
// strict-rune validation. It's already used implicitly via range over strings,
// but keeping the import documents the intent.
const _ = utf8.RuneError
