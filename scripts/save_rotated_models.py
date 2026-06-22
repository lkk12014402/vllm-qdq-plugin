#!/usr/bin/env python3
"""
Save rotated + quantized models for evaluation.

Exports Qwen3-0.6B with different rotation configurations:

  vLLM + HF supported:
    - R1 only (default online Hadamard)
    - R1+R2 (R2 offline fused into head)
    - R1+R4 (R4 online on down_proj)
    - R1+R2+R4 (R2 fused + R4 online)
    - R1 with custom rotation_size=128
    - R1 with random Hadamard
    - R1+R4 with custom rotation_size=128

  HF only (vLLM R3 not yet implemented):
    - R1+R2+R3 (R3 online Q/K rotation)
    - R1+R2+R3+R4 (all rotations)

All use MXFP4 quantization, iters=0 (RTN, no tuning).

Usage:
    # Save all configs:
    python save_rotated_models.py --device cuda:7

    # Save specific config:
    python save_rotated_models.py --device cuda:7 --configs R1 R1+R4

    # With custom base output dir:
    python save_rotated_models.py --device cuda:7 --output-base ./rotated_models
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto-round"))

import torch

from auto_round import AutoRound
from auto_round.algorithms.transforms.spinquant import SpinQuantConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Rotation configurations
# ═══════════════════════════════════════════════════════════════════════════════

# Configs supported by BOTH HF and vLLM plugin
ROTATION_CONFIGS = {
    "R1": dict(
        r1=True, r2=False, r3=False, r4=False,
        online_r1_rotation=True,
        rotation_size=None,
        random_r1=False,
    ),
    "R1+R2": dict(
        r1=True, r2=True, r3=False, r4=False,
        online_r1_rotation=True,
        rotation_size=None,
        random_r1=False,
    ),
    "R1+R4": dict(
        r1=True, r2=False, r3=False, r4=True,
        online_r1_rotation=True,
        rotation_size=None,
        random_r1=False,
    ),
    "R1+R2+R4": dict(
        r1=True, r2=True, r3=False, r4=True,
        online_r1_rotation=True,
        rotation_size=None,
        random_r1=False,
    ),
    "R1_size128": dict(
        r1=True, r2=False, r3=False, r4=False,
        online_r1_rotation=True,
        rotation_size=128,
        random_r1=False,
    ),
    "R1_size32": dict(
        r1=True, r2=False, r3=False, r4=False,
        online_r1_rotation=True,
        rotation_size=32,
        random_r1=False,
    ),
    "R1_random": dict(
        r1=True, r2=False, r3=False, r4=False,
        online_r1_rotation=True,
        rotation_size=None,
        random_r1=True,
    ),
    "R1+R4_size128": dict(
        r1=True, r2=False, r3=False, r4=True,
        online_r1_rotation=True,
        rotation_size=128,
        random_r1=False,
    ),
    "R1+R4_size32": dict(
        r1=True, r2=False, r3=False, r4=True,
        online_r1_rotation=True,
        rotation_size=32,
        random_r1=False,
    ),
    "R1+R2+R4_size128": dict(
        r1=True, r2=True, r3=False, r4=True,
        online_r1_rotation=True,
        rotation_size=128,
        random_r1=False,
    ),
    "R1+R2+R4_size32": dict(
        r1=True, r2=True, r3=False, r4=True,
        online_r1_rotation=True,
        rotation_size=32,
        random_r1=False,
    ),
    "R1+R2_size32": dict(
        r1=True, r2=True, r3=False, r4=False,
        online_r1_rotation=True,
        rotation_size=32,
        random_r1=False,
    ),
    "R1+R2_size128": dict(
        r1=True, r2=True, r3=False, r4=False,
        online_r1_rotation=True,
        rotation_size=128,
        random_r1=False,
    ),
}

# Configs with R3 — HF eval only (vLLM plugin does NOT support R3 yet)
ROTATION_CONFIGS_HF_ONLY = {
    "R1+R2+R3": dict(
        r1=True, r2=True, r3=True, r4=False,
        online_r1_rotation=True,
        rotation_size=None,
        random_r1=False,
    ),
    "R1+R2+R3+R4": dict(
        r1=True, r2=True, r3=True, r4=True,
        online_r1_rotation=True,
        rotation_size=None,
        random_r1=False,
    ),
}

# Combined for convenience
ALL_ROTATION_CONFIGS = {**ROTATION_CONFIGS, **ROTATION_CONFIGS_HF_ONLY}


def save_model(
    model_name: str,
    config_name: str,
    config_kwargs: dict,
    output_base: str,
    device: str,
    scheme: str = "MXFP4",
    nsamples: int = 128,
    seqlen: int = 512,
):
    """Export one rotated model."""
    output_dir = os.path.join(output_base, f"{config_name}")
    if os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"  ⏭️  {config_name} already exists at {output_dir}, skipping")
        return output_dir

    print(f"\n{'═'*70}")
    print(f"  Saving: {config_name}")
    print(f"  Config: {config_kwargs}")
    print(f"  Output: {output_dir}")
    print(f"{'═'*70}")

    rotation_config = SpinQuantConfig(
        **config_kwargs,
        trainable_rotation=False,
        trainable_smooth=False,
    )

    t0 = time.time()
    ar = AutoRound(
        model_name,
        rotation_config=rotation_config,
        scheme=scheme,
        iters=0,
        #iters=200,
        nsamples=nsamples,
        seqlen=seqlen,
        device_map="auto",
    )
    ar.quantize_and_save(output_dir=output_dir, format="auto_round")

    elapsed = time.time() - t0

    # Find actual model subdirectory (AutoRound creates e.g. Qwen3-0.6B-mxfp-w4g32/)
    actual_model_dir = output_dir
    for sub in os.listdir(output_dir):
        sub_path = os.path.join(output_dir, sub)
        if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "config.json")):
            actual_model_dir = sub_path
            break

    print(f"  ✅ Saved in {elapsed:.1f}s → {actual_model_dir}")

    # Cleanup
    del ar
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Save rotated+quantized models")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--output-base", default=None,
                        help="Base output dir (default: ./rotated_models_<model_short>)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help=f"Which configs to save. Options: {list(ROTATION_CONFIGS.keys())}")
    parser.add_argument("--scheme", default="MXFP4")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    if args.output_base is None:
        args.output_base = f"./rotated_models_{model_short}"

    configs_to_run = args.configs or list(ALL_ROTATION_CONFIGS.keys())
    invalid = [c for c in configs_to_run if c not in ALL_ROTATION_CONFIGS]
    if invalid:
        print(f"ERROR: Unknown configs: {invalid}")
        print(f"Available: {list(ALL_ROTATION_CONFIGS.keys())}")
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Scheme: {args.scheme}")
    print(f"Output base: {args.output_base}")
    print(f"Configs to save: {configs_to_run}")
    print(f"Device: {args.device}")

    os.makedirs(args.output_base, exist_ok=True)
    saved_dirs = {}

    for config_name in configs_to_run:
        config_kwargs = ALL_ROTATION_CONFIGS[config_name]
        output_dir = save_model(
            args.model, config_name, config_kwargs,
            args.output_base, args.device,
            scheme=args.scheme,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
        )
        saved_dirs[config_name] = output_dir

    # Write a summary file
    summary = {
        "model": args.model,
        "scheme": args.scheme,
        "configs": {},
    }
    for k, v in saved_dirs.items():
        entry = {"path": v, **ALL_ROTATION_CONFIGS[k]}
        entry["vllm_supported"] = k not in ROTATION_CONFIGS_HF_ONLY
        summary["configs"][k] = entry

    summary_path = os.path.join(args.output_base, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'═'*70}")
    print(f"  ALL DONE — {len(saved_dirs)} models saved")
    print(f"  Summary: {summary_path}")
    print(f"{'═'*70}")
    for name, path in saved_dirs.items():
        print(f"  {name:20s} → {path}")


if __name__ == "__main__":
    main()
