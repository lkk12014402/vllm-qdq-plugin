#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""vLLM-backend smoke test for the per-Linear / random Hadamard MXFP4 plugin.

Requires the vllm-qdq-plugin installed (``pip install -e vllm-qdq-plugin``) and
``VLLM_HADAMARD_MXFP4=1`` so the ``hadamard_mxfp4`` quantization config is
registered and auto-detected from the checkpoint's ``rotation_config``.

Usage:
    CUDA_VISIBLE_DEVICES=7 VLLM_HADAMARD_MXFP4=1 \
        python smoke_vllm_hadamard.py <model_dir>
"""

from __future__ import annotations

import sys

from vllm import LLM, SamplingParams


def main(model_dir: str) -> None:
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=True,
        trust_remote_code=True,
    )
    prompts = [
        "What is the capital of France? Answer in one word.",
        "Write a one-sentence summary of what a neural network is.",
        "2 + 2 * 3 = ?",
    ]
    msgs = [[{"role": "user", "content": p}] for p in prompts]
    out = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=48))
    for p, o in zip(prompts, out):
        print("=" * 60)
        print("PROMPT:", p)
        print("OUTPUT:", o.outputs[0].text.strip())


if __name__ == "__main__":
    main(sys.argv[1])
