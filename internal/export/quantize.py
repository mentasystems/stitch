"""Quantize needle.pkl's Dense kernels to int8 (per-column symmetric).

For each Dense kernel tensor of shape (..., in_dim, out_dim) we compute:

    scales[..., j] = max(|W[..., :, j]|) / 127
    W_int8[..., i, j] = round(W[..., i, j] / scales[..., j])

then store W_int8 as dtype "int8" and an accompanying "<name>_scale" tensor
as fp16. Norms, gates, and embeddings stay fp16 to preserve quality.

Output is the same STITCH01 file format as export.py.

Run:
    .venv/bin/python internal/export/quantize.py \\
        --checkpoint /Users/jairo/needle/checkpoints/needle.pkl \\
        --out internal/weights/weights_int8.bin
"""
import argparse
import json
import os
import pickle
import struct

import numpy as np


# Kernel substrings whose tensors we should quantize. Everything else stays
# fp16. (We skip the embedding because it's also used as the output projection,
# which is the most sensitive op for greedy decoding.)
KERNEL_KEYWORDS = ("q_proj/kernel", "k_proj/kernel", "v_proj/kernel", "out_proj/kernel")


def flatten(tree, prefix=""):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(flatten(v, prefix + ("/" if prefix else "") + k))
    else:
        out[prefix] = tree
    return out


def should_quantize(name):
    return any(kw in name for kw in KERNEL_KEYWORDS)


def quantize_per_col(w):
    """Symmetric per-output-column int8 quantization.

    w shape: (..., in_dim, out_dim). Scales shape: (..., out_dim).
    Returns (w_int8, scales_fp16) with reproducible round-half-to-even.
    """
    # max along the in_dim axis (second-to-last)
    max_abs = np.maximum(np.max(np.abs(w), axis=-2), 1e-8)
    scales = (max_abs / 127.0).astype(np.float32)
    # Broadcast scales for division: (..., 1, out_dim)
    inv = (1.0 / scales)[..., None, :]
    quant = np.round(w.astype(np.float32) * inv)
    quant = np.clip(quant, -127, 127).astype(np.int8)
    return quant, scales.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    config = dict(ckpt["config"])
    flat = flatten(ckpt["params"])

    drop_prefixes = ("contrastive_hidden", "contrastive_proj", "log_temp")
    flat = {k: v for k, v in flat.items() if not k.lstrip("/").startswith(drop_prefixes)}

    tensors = {}
    blob_chunks = []
    offset = 0
    total_int8 = 0
    total_fp16 = 0

    for name in sorted(flat.keys()):
        clean = name.lstrip("/")
        w = np.asarray(flat[name])
        if should_quantize(clean):
            q, sc = quantize_per_col(w)
            q_bytes = np.ascontiguousarray(q, dtype=np.int8).tobytes()
            tensors[clean] = {
                "shape": list(q.shape),
                "dtype": "int8",
                "offset": offset,
                "nbytes": len(q_bytes),
            }
            blob_chunks.append(q_bytes)
            offset += len(q_bytes)
            total_int8 += q.size

            sc_bytes = np.ascontiguousarray(sc, dtype="<f2").tobytes()
            sc_name = clean + "_scale"
            tensors[sc_name] = {
                "shape": list(sc.shape),
                "dtype": "fp16",
                "offset": offset,
                "nbytes": len(sc_bytes),
            }
            blob_chunks.append(sc_bytes)
            offset += len(sc_bytes)
            total_fp16 += sc.size
        else:
            arr = np.ascontiguousarray(w, dtype="<f2")
            data = arr.tobytes()
            tensors[clean] = {
                "shape": list(arr.shape),
                "dtype": "fp16",
                "offset": offset,
                "nbytes": len(data),
            }
            blob_chunks.append(data)
            offset += len(data)
            total_fp16 += arr.size

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
    print(f"  int8 params:  {total_int8:>12,}")
    print(f"  fp16 params:  {total_fp16:>12,}")
    print(f"  header:       {len(header_bytes):,} bytes")
    print(f"  blob:         {offset:,} bytes")
    print(f"  total:        {total_bytes:,} bytes ({total_bytes/1024/1024:.1f} MiB)")


if __name__ == "__main__":
    main()
