#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end HF vs vLLM comparison for the per-Linear Hadamard MXFP4 plugin.

Runs the same prompts through HF (auto-round convert_model + rotation hooks) and
vLLM (vllm-qdq-plugin ``hadamard_mxfp4``) with greedy decoding and reports
whether the generated token strings agree.

Usage:
    CUDA_VISIBLE_DEVICES=7 VLLM_HADAMARD_MXFP4=1 \
        python compare_hf_vllm_hadamard.py <model_dir>
"""

from __future__ import annotations

import gc
import sys

import torch

PROMPTS = [
    "What is the capital of France? Answer with just the city name.",
    "Name three primary colors.",
    "What is 12 multiplied by 8?",
]
MAXTOK = 200


def build_inputs(tok):
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in PROMPTS
    ]


def run_hf(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype="bfloat16", device_map="cuda:0", trust_remote_code=True
    )
    outs = []
    for text in build_inputs(tok):
        ids = tok(text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=MAXTOK, do_sample=False)
        outs.append(tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outs


def run_vllm(model_dir):
    from vllm import LLM, SamplingParams

    tok_texts = None
    llm = LLM(
        model=model_dir, dtype="bfloat16", gpu_memory_utilization=0.85,
        max_model_len=4096, enforce_eager=True, trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    tok_texts = build_inputs(tok)
    out = llm.generate(tok_texts, SamplingParams(temperature=0.0, max_tokens=MAXTOK))
    return [o.outputs[0].text for o in out]


def main(model_dir):
    mode = sys.argv[2] if len(sys.argv) > 2 else "both"
    if mode == "hf":
        for p, o in zip(PROMPTS, run_hf(model_dir)):
            print("=" * 60, "\nPROMPT:", p, "\nHF:", o.strip())
        return
    if mode == "vllm":
        for p, o in zip(PROMPTS, run_vllm(model_dir)):
            print("=" * 60, "\nPROMPT:", p, "\nVLLM:", o.strip())
        return
    hf = run_hf(model_dir)
    vl = run_vllm(model_dir)
    match = 0
    for p, h, v in zip(PROMPTS, hf, vl):
        same = h.strip() == v.strip()
        match += same
        print("=" * 70)
        print("PROMPT:", p)
        print("HF  :", h.strip()[:300])
        print("VLLM:", v.strip()[:300])
        print("EXACT MATCH:", same)
    print("\nSUMMARY: %d/%d prompts exact-match" % (match, len(PROMPTS)))


if __name__ == "__main__":
    main(sys.argv[1])
