"""Convert needle.pkl (JAX/Flax) into stitch's flat binary format.

Output format (little-endian):
    magic:       8 bytes "STITCH01"
    header_len:  4 bytes uint32
    header:      JSON {"config": {...}, "tensors": {name: {shape, dtype, offset, nbytes}}}
    blob:        concatenated fp16 tensor bytes

Run from inside the needle venv:
    .venv/bin/python internal/export/export.py \\
        --checkpoint /Users/jairo/needle/checkpoints/needle.pkl \\
        --out internal/weights/weights.bin
"""
import argparse
import json
import os
import pickle
import struct

import numpy as np


def flatten(tree, prefix=""):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(flatten(v, prefix + ("/" if prefix else "") + k))
    else:
        out[prefix] = tree
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)

    config = dict(ckpt["config"])
    flat = flatten(ckpt["params"])

    # Drop weights not needed for tool-call generation.
    drop_prefixes = ("contrastive_hidden", "contrastive_proj", "log_temp")
    flat = {k: v for k, v in flat.items() if not k.lstrip("/").startswith(drop_prefixes)}

    tensors = {}
    blob_chunks = []
    offset = 0
    total_params = 0
    for name in sorted(flat.keys()):
        arr = np.asarray(flat[name]).astype(np.float16)
        # Force contiguous little-endian fp16
        arr = np.ascontiguousarray(arr, dtype="<f2")
        data = arr.tobytes()
        tensors[name.lstrip("/")] = {
            "shape": list(arr.shape),
            "dtype": "fp16",
            "offset": offset,
            "nbytes": len(data),
        }
        blob_chunks.append(data)
        offset += len(data)
        total_params += int(np.prod(arr.shape))

    header = {"config": config, "tensors": tensors}
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b"STITCH01")
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        for chunk in blob_chunks:
            f.write(chunk)

    total_bytes = os.path.getsize(args.out)
    print(f"Wrote {args.out}")
    print(f"  params:  {total_params:,}")
    print(f"  tensors: {len(tensors)}")
    print(f"  header:  {len(header_bytes):,} bytes")
    print(f"  blob:    {offset:,} bytes")
    print(f"  total:   {total_bytes:,} bytes ({total_bytes/1024/1024:.1f} MiB)")


if __name__ == "__main__":
    main()
