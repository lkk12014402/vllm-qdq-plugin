from __future__ import annotations

import argparse
import gc
import sys
import os
import time


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_multi_gpu(device: str) -> bool:
    """Check if device spec indicates multi-GPU usage.

    Multi-GPU formats:
      - "auto"       — accelerate auto device_map
      - "0,1,2,3"   — explicit GPU list
      - "0,1"       — two GPUs
    Single-GPU formats:
      - "cuda:0"    — single GPU
      - "cuda"      — default GPU
      - "cpu"       — CPU only
    """
    if device is None:
        return False
    device = str(device).strip()
    if device == "auto":
        return True
    if "," in device:
        return True
    return False


def get_primary_device(device: str) -> str:
    """Get the primary device for single-tensor operations.

    For multi-GPU: returns "cuda:0" (first GPU).
    For single-GPU: returns the device as-is.
    """
    if not is_multi_gpu(device):
        return device
    device = str(device).strip()
    if device == "auto":
        return "cuda:0"
    # "0,1,2,3" → "cuda:0"
    first_gpu = device.split(",")[0].strip()
    return f"cuda:{first_gpu}"


def evaluate_model(model_path, tasks="gsm8k,mmlu,hellaswag,piqa", batch_size=32, limit=None, device="cuda:0"):
    """Common evaluation via lm_eval (matches the other scripts in this dir)."""
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    multi_gpu = is_multi_gpu(device)
    primary_dev = get_primary_device(device)

    common_kwargs = dict(
        pretrained=model_path, batch_size=batch_size,
        dtype="bfloat16", trust_remote_code=True,
        add_bos_token=True, softmax_dtype="float32",
    )
    if multi_gpu:
        lm = HFLM(**common_kwargs, parallelize=True)
    else:
        lm = HFLM(**common_kwargs, device=primary_dev)

    task_list = [t.strip() for t in tasks.split(",")] if isinstance(tasks, str) else tasks
    results = simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=batch_size,
        limit=limit,
        gen_kwargs="max_gen_toks=2048",
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        fewshot_random_seed=42)
    metrics = {}
    print(results.get("results", {}))
    for task_name, task_results in results.get("results", {}).items():
        acc = task_results.get("acc,none") or task_results.get("acc_norm,none")
        if acc is not None:
            metrics[task_name] = round(acc, 4)
    return metrics


#model_path = "Qwen3-8_autoround_hadamard_rtn_mxfp4_main/Qwen3-8B-mxfp-w4g32/"
#model_path = "./Qwen3-8_autoround_hadamard_rtn_mxfp4_old/"
#model_path = "Qwen3-8B-rotated-r1r2r3r4-128-mxfp4-iters0/R1+R2+R3+R4/Qwen3-8B-mxfp-w4g32/"
model_path = "Qwen3-8_autoround_hadamard_rtn_mxfp4/Qwen3-8B-mxfp-w4g32/"
model_path = "Qwen3-8_autoround_random_hadamard_rtn_mxfp4/Qwen3-8B-mxfp-w4g32/"
model_path = "./rotated_models_Qwen3-8B/R1+R4/Qwen3-8B-mxfp-w4g32"

#tasks = "piqa"
tasks = "piqa"

evaluate_model(model_path, tasks, device="auto")
