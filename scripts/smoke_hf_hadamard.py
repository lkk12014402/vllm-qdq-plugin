#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HF-backend smoke test for an auto-round per-Linear / random Hadamard MXFP4 model.

Loads the model via transformers (which routes through auto_round's
`convert_model.py` -> `apply_rotation_hooks_from_config`, registering one forward
pre-hook per quantized Linear), then runs a short greedy generation.

Usage:
    CUDA_VISIBLE_DEVICES=7 python smoke_hf_hadamard.py <model_dir>

Confirms:
  * rotation forward pre-hooks are registered (count == #quantized Linears)
  * generation produces coherent text
"""

from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(model_dir: str) -> None:
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype="bfloat16", device_map="cuda:0", trust_remote_code=True
    )

    n_hooks = sum(len(m._forward_pre_hooks) for m in model.modules())
    print("registered forward_pre_hooks (rotation hooks):", n_hooks)

    msg = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    enc = tok.apply_chat_template(
        msg, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda:0")
    out = model.generate(enc["input_ids"], max_new_tokens=20, do_sample=False)
    print("OUTPUT:", repr(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)))


if __name__ == "__main__":
    main(sys.argv[1])
