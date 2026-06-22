#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inspect an auto-round per-Linear / random Hadamard MXFP4 checkpoint.

Usage:
    python inspect_hadamard_model.py <model_dir>

Reports:
  * quantization_config.rotation_config (hadamard_type / block_size / backend)
  * whether per-Linear `hadamard_matrix` buffers are serialized
  * how many distinct Hadamard matrices exist (shared vs per-Linear)
  * orthonormality + scale of the stored 32x32 matrix

Findings on the reference Qwen3-8B checkpoints (2026-06):
  * random_hadamard  -> 252 `hadamard_matrix` buffers, ALL IDENTICAL
                        (one shared random Hadamard, orthonormal, |v|=1/sqrt(32))
  * hadamard (det.)  -> 0  `hadamard_matrix` buffers (regenerated at load time)
"""

from __future__ import annotations

import glob
import json
import os
import sys

import torch
from safetensors import safe_open


def main(model_dir: str) -> None:
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    qc = cfg.get("quantization_config", {})
    print("=== quantization_config ===")
    print("  data_type     :", qc.get("data_type"), "bits:", qc.get("bits"), "group_size:", qc.get("group_size"))
    print("  act_data_type :", qc.get("act_data_type"), "act_bits:", qc.get("act_bits"),
          "act_group_size:", qc.get("act_group_size"))
    print("  rotation_config:", qc.get("rotation_config"))

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        wmap = json.load(open(index_path))["weight_map"]
        keys = list(wmap)
    else:
        keys = []
        for shard in glob.glob(os.path.join(model_dir, "*.safetensors")):
            with safe_open(shard, "pt") as f:
                keys.extend(f.keys())

    had_keys = [k for k in keys if k.endswith("hadamard_matrix")]
    print("\n=== hadamard_matrix buffers ===")
    print("  count:", len(had_keys))
    if not had_keys:
        print("  -> deterministic Hadamard: regenerated at load time (Sylvester), not stored.")
        return

    mats = {}
    for shard in glob.glob(os.path.join(model_dir, "*.safetensors")):
        with safe_open(shard, "pt") as f:
            for k in f.keys():
                if k.endswith("hadamard_matrix"):
                    mats[k] = f.get_tensor(k).float()

    ref = next(iter(mats.values()))
    all_same = all(torch.equal(ref, m) for m in mats.values())
    print("  shape:", tuple(ref.shape), "dtype: float32")
    print("  ALL identical (shared random Hadamard):", all_same)
    I = ref @ ref.T
    print("  orthonormal (H@H.T == I):",
          torch.allclose(I, torch.eye(ref.shape[0]), atol=1e-4),
          "| max off-diag:", (I - torch.eye(ref.shape[0])).abs().max().item())
    print("  unique |values|:", torch.unique(ref.abs()).tolist()[:4],
          "| 1/sqrt(N):", 1.0 / (ref.shape[0] ** 0.5))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
