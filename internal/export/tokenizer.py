"""Convert SentencePiece .model into a simple format for Go.

Output: internal/weights/tokenizer.bin
    8 bytes:  magic "STITCK01"
    4 bytes:  uint32 vocab_size
    1 byte:   byte_fallback (0/1)
    1 byte:   add_dummy_prefix (0/1)  — sentencepiece always prepends ▁ to input
    1 byte:   reserved
    1 byte:   reserved
    For each piece (vocab_size entries, id-indexed):
        4 bytes: float32 score
        1 byte:  type (0=NORMAL,1=UNKNOWN,2=CONTROL,3=USER_DEFINED,4=BYTE,5=UNUSED)
        2 bytes: uint16 piece byte length
        N bytes: piece UTF-8 bytes
"""
import argparse
import os
import struct

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as model_pb2


# Map proto Type -> our flat byte.
TYPE_MAP = {
    model_pb2.ModelProto.SentencePiece.NORMAL: 0,
    model_pb2.ModelProto.SentencePiece.UNKNOWN: 1,
    model_pb2.ModelProto.SentencePiece.CONTROL: 2,
    model_pb2.ModelProto.SentencePiece.USER_DEFINED: 3,
    model_pb2.ModelProto.SentencePiece.BYTE: 4,
    model_pb2.ModelProto.SentencePiece.UNUSED: 5,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proto = model_pb2.ModelProto()
    with open(args.model, "rb") as f:
        proto.ParseFromString(f.read())

    pieces = proto.pieces
    vocab_size = len(pieces)
    byte_fallback = 1 if proto.trainer_spec.byte_fallback else 0
    add_dummy_prefix = 1 if proto.normalizer_spec.add_dummy_prefix else 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"STITCK01")
        f.write(struct.pack("<I", vocab_size))
        f.write(bytes([byte_fallback, add_dummy_prefix, 0, 0]))
        for piece in pieces:
            piece_bytes = piece.piece.encode("utf-8")
            f.write(struct.pack("<f", piece.score))
            f.write(bytes([TYPE_MAP[piece.type]]))
            f.write(struct.pack("<H", len(piece_bytes)))
            f.write(piece_bytes)

    # Sanity check: reload with sentencepiece and verify a few encodings.
    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)
    for text in [
        "What's the weather in San Francisco?",
        '<tools>[{"name":"get_weather"}]',
        "hello world",
    ]:
        ids = sp.Encode(text, out_type=int)
        pieces_dec = [sp.IdToPiece(i) for i in ids]
        print(f"{text!r} -> {ids}")
        print(f"  pieces: {pieces_dec}")

    print(f"\nWrote {args.out}")
    print(f"  vocab_size:       {vocab_size}")
    print(f"  byte_fallback:    {byte_fallback}")
    print(f"  add_dummy_prefix: {add_dummy_prefix}")
    print(f"  size:             {os.path.getsize(args.out):,} bytes")


if __name__ == "__main__":
    main()
