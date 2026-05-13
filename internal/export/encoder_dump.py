"""Dump encoder_out for the golden query so Go can compare element-wise.

Writes internal/testdata/encoder_out.json:
    enc_tokens, shape, encoder_out (fp32 list flattened row-major)
"""
import json
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/Users/jairo/needle")
from needle.model.architecture import SimpleAttentionNetwork, make_padding_mask
from needle.model.run import _build_encoder_input, load_checkpoint, DEFAULT_MAX_ENC_LEN
from needle.dataset.dataset import get_tokenizer


def main():
    params, config = load_checkpoint("/Users/jairo/needle/checkpoints/needle.pkl")
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer()

    query = "What's the weather in San Francisco?"
    tools = '[{"name":"get_weather","parameters":{"location":"string"}}]'
    enc_tokens = _build_encoder_input(tokenizer, query, tools, DEFAULT_MAX_ENC_LEN)
    enc_input = jnp.array([enc_tokens])
    src_mask = make_padding_mask(enc_input, tokenizer.pad_token_id)
    encoder_out, _ = model.apply(
        {"params": params}, enc_input, src_mask=src_mask, method="encode"
    )
    arr = np.asarray(encoder_out, dtype=np.float32)[0]  # drop batch
    out = {
        "enc_tokens": list(enc_tokens),
        "shape": list(arr.shape),
        "encoder_out": arr.flatten().tolist(),
    }
    os.makedirs("internal/testdata", exist_ok=True)
    with open("internal/testdata/encoder_out.json", "w") as f:
        json.dump(out, f)
    print(f"Wrote encoder_out.json: shape {arr.shape}, range [{arr.min():.4f}, {arr.max():.4f}]")


if __name__ == "__main__":
    main()
