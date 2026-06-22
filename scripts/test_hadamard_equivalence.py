#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-Linear equivalence: vLLM Hadamard-MXFP4 LinearMethod vs auto-round HF.

Loads the auto-round random/deterministic Hadamard MXFP4 model under HF (which
registers the per-Linear inverse-block-Hadamard forward pre-hooks), captures the
RAW input and reference output of a chosen Linear, then runs the standalone
``HadamardMXFP4LinearMethod`` forward on the same raw input and compares.

A close match (only MXFP4 act-qdq rounding differs, which is shared logic)
validates that the migrated vLLM plugin reproduces auto-round's math.

Usage:
    CUDA_VISIBLE_DEVICES=7 python test_hadamard_equivalence.py <model_dir> [--type random|hadamard]
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "vllm-qdq-plugin/src")
from vllm_qdq_plugin.rotation.mxfp4 import register_custom_ops  # noqa: E402
from vllm_qdq_plugin.rotation.perlinear_config import HadamardMXFP4Config  # noqa: E402
from vllm_qdq_plugin.rotation.perlinear_linear_method import (  # noqa: E402
    HadamardMXFP4LinearMethod,
)

TARGET = "model.layers.0.mlp.gate_proj"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--type", default="random_hadamard")
    args = ap.parse_args()

    register_custom_ops()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype="bfloat16", device_map="cuda:0", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    target_mod = dict(model.named_modules())[TARGET]

    captured = {}

    def pre_capture(mod, args_in):  # runs FIRST (prepend) -> raw input
        captured["raw_x"] = args_in[0].detach().clone()
        return None

    def post_capture(mod, args_in, output):
        captured["y_ref"] = output.detach().clone()

    h1 = target_mod.register_forward_pre_hook(pre_capture, prepend=True)
    h2 = target_mod.register_forward_hook(post_capture)

    msg = [{"role": "user", "content": "Hello, briefly introduce yourself."}]
    enc = tok.apply_chat_template(
        msg, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda:0")
    with torch.no_grad():
        model(**enc)
    h1.remove()
    h2.remove()

    raw_x = captured["raw_x"]
    y_ref = captured["y_ref"]

    # Build the vLLM LinearMethod layer state directly from the HF qmodule tensors.
    cfg = HadamardMXFP4Config(hadamard_type=args.type, block_size=32, group_size=32)
    method = HadamardMXFP4LinearMethod(cfg)

    layer = torch.nn.Module()
    layer.params_dtype = torch.bfloat16
    # Single (unmerged) Linear -> one output partition.
    layer._output_partition_sizes = [target_mod.weight_packed.shape[0]]
    layer.weight_packed = torch.nn.Parameter(target_mod.weight_packed.data, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(target_mod.weight_scale.data, requires_grad=False)
    if args.type == "random_hadamard":
        Hbuf = target_mod.hadamard_matrix.data.float().reshape(1, 32, 32)
    else:
        Hbuf = torch.eye(32).reshape(1, 32, 32)  # unused for deterministic (regenerated)
    layer.hadamard_matrix = torch.nn.Parameter(Hbuf, requires_grad=False)
    method.process_weights_after_loading(layer)

    with torch.no_grad():
        y_plugin = method.apply(layer, raw_x.to("cuda:0"))

    y_ref = y_ref.float()
    y_plugin = y_plugin.float()
    abs_err = (y_plugin - y_ref).abs()
    denom = y_ref.abs().mean().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(
        y_plugin.reshape(-1), y_ref.reshape(-1), dim=0
    )
    print(f"target            : {TARGET}")
    print(f"hadamard_type     : {args.type}")
    print(f"raw_x shape       : {tuple(raw_x.shape)}")
    print(f"max abs err       : {abs_err.max().item():.6e}")
    print(f"mean abs err      : {abs_err.mean().item():.6e}")
    print(f"rel mean err      : {(abs_err.mean()/denom).item():.6e}")
    print(f"cosine similarity : {cos.item():.8f}")
    # The HF triton reference itself rotates in bf16 (tl.dot); this plugin rotates
    # in fp32, so a ~1% mean-relative residual (amplified by near-zero outputs) is
    # expected and benign. Cosine similarity is the robust correctness metric.
    ok = cos.item() > 0.9995
    print("RESULT            :", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
