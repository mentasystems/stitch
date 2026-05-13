"""Capture golden inputs/outputs from the JAX reference for Go verification.

Dumps to internal/testdata/golden.json:
    query
    tools
    enc_tokens         (int list)
    encoder_out_md5    (md5 of fp16 bytes of encoder_out)
    encoder_out_sample (first 16 floats of encoder_out[0,0,:])
    generated_tokens   (int list, no leading EOS, no trailing EOS)
    decoded            (str — final tool-call JSON output)
"""
import hashlib
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/Users/jairo/needle")
from needle.model.architecture import (
    SimpleAttentionNetwork,
    make_padding_mask,
)
from needle.model.run import (
    _build_encoder_input,
    load_checkpoint,
    DEFAULT_MAX_ENC_LEN,
)
from needle.dataset.dataset import get_tokenizer


def main():
    out_path = "internal/testdata/golden.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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

    enc_np = np.asarray(encoder_out).astype(np.float16)
    md5 = hashlib.md5(enc_np.tobytes()).hexdigest()
    sample = enc_np[0, 0, :16].astype(float).tolist()

    # Capture decoder logits at step 0 (greedy prediction for position-after-EOS).
    from needle.model.architecture import make_causal_mask
    max_gen_len = 64
    dec_buffer = jnp.full((1, max_gen_len), tokenizer.pad_token_id, dtype=jnp.int32)
    dec_buffer = dec_buffer.at[0, 0].set(tokenizer.eos_token_id)
    tgt_mask = make_causal_mask(max_gen_len)

    logits = model.apply(
        {"params": params}, dec_buffer, encoder_out,
        self_mask=tgt_mask, cross_mask=None, method="decode",
    )
    step0_logits = np.asarray(logits[0, 0]).astype(float)
    step0_top5 = sorted(enumerate(step0_logits.tolist()), key=lambda x: -x[1])[:5]

    # Full greedy generate (no constrained decoder, to keep this self-contained).
    generated = []
    for i in range(max_gen_len - 1):
        logits = model.apply(
            {"params": params}, dec_buffer, encoder_out,
            self_mask=tgt_mask, cross_mask=None, method="decode",
        )
        next_token = int(jnp.argmax(logits[0, i]))
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)
        dec_buffer = dec_buffer.at[0, i + 1].set(next_token)

    decoded = tokenizer.decode(generated)

    golden = {
        "query": query,
        "tools": tools,
        "enc_tokens": list(enc_tokens),
        "encoder_out_shape": list(enc_np.shape),
        "encoder_out_md5": md5,
        "encoder_out_sample_first16": sample,
        "step0_top5": step0_top5,
        "generated_tokens": generated,
        "decoded": decoded,
        "pad_id": tokenizer.pad_token_id,
        "eos_id": tokenizer.eos_token_id,
        "bos_id": tokenizer.bos_token_id,
        "tools_sep_id": tokenizer.tools_token_id,
        "tool_call_id": tokenizer.tool_call_token_id,
        "vocab_size": tokenizer.vocab_size,
    }
    with open(out_path, "w") as f:
        json.dump(golden, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"  enc_tokens: {len(enc_tokens)}")
    print(f"  encoder_out shape: {enc_np.shape}")
    print(f"  generated tokens: {len(generated)}")
    print(f"  decoded: {decoded!r}")


if __name__ == "__main__":
    main()
