#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate rotated models with lm_eval using vLLM backend.
#
# This evaluates each model using vLLM with the spinquant_mxfp4 plugin.
# The plugin handles online rotation (R1, R4) during inference.
#
# Prerequisites:
#   - vLLM installed with the auto_round plugin registered
#   - Models saved by save_rotated_models.py
#
# Usage:
#   # Evaluate all saved models:
#   bash eval_vllm.sh
#
#   # Evaluate specific model:
#   bash eval_vllm.sh ./rotated_models_Qwen3-0.6B/R1+R4
#
#   # Change GPU/tasks/limit:
#   CUDA_VISIBLE_DEVICES=6 TASKS="piqa,hellaswag" LIMIT=50 bash eval_vllm.sh
#
#   # Multi-GPU (tensor parallel):
#   CUDA_VISIBLE_DEVICES=4,5 NUM_GPUS=2 bash eval_vllm.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -e

# Configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

NUM_GPUS=${NUM_GPUS:-1}
TASKS=${TASKS:-"piqa"}
BATCH_SIZE=${BATCH_SIZE:-8}
LIMIT=${LIMIT:-100}
OUTPUT_BASE=${OUTPUT_BASE:-"./lm_eval_results_vllm"}
MODEL_BASE=${MODEL_BASE:-"./rotated_models_Qwen3-8B"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

mkdir -p "$OUTPUT_BASE"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  lm_eval — vLLM backend (with spinquant_mxfp4 plugin)"
echo "  Tasks: $TASKS"
echo "  Limit: $LIMIT"
echo "  Batch size: $BATCH_SIZE"
echo "  GPUs: $CUDA_VISIBLE_DEVICES (TP=$NUM_GPUS)"
echo "  Model base: $MODEL_BASE"
echo "═══════════════════════════════════════════════════════════════════════"

evaluate_model() {
    local MODEL_PATH="$1"
    local CONFIG_NAME="$2"
    local RESULT_DIR="${OUTPUT_BASE}/${CONFIG_NAME}"

    # Skip R3-containing configs (not supported by vLLM plugin yet)
    if echo "$CONFIG_NAME" | grep -q "R3"; then
        echo ""
        echo "  ⏭️  Skipping $CONFIG_NAME (R3 not supported in vLLM plugin)"
        return
    fi

    echo ""
    echo "───────────────────────────────────────────────────────────────────────"
    echo "  Evaluating: $CONFIG_NAME (vLLM)"
    echo "  Model path: $MODEL_PATH"
    echo "  Results: $RESULT_DIR"
    echo "───────────────────────────────────────────────────────────────────────"

    if [ -f "${RESULT_DIR}/results.json" ]; then
        echo "  ⏭️  Results already exist, skipping"
        return
    fi

    mkdir -p "$RESULT_DIR"

    MODEL_ARGS="pretrained=${MODEL_PATH}"
    MODEL_ARGS="${MODEL_ARGS},tensor_parallel_size=${NUM_GPUS}"
    MODEL_ARGS="${MODEL_ARGS},max_model_len=${MAX_MODEL_LEN}"
    MODEL_ARGS="${MODEL_ARGS},max_num_batched_tokens=32768"
    MODEL_ARGS="${MODEL_ARGS},max_num_seqs=128"
    MODEL_ARGS="${MODEL_ARGS},add_bos_token=True"
    MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    MODEL_ARGS="${MODEL_ARGS},dtype=bfloat16"
    MODEL_ARGS="${MODEL_ARGS},max_gen_toks=2048"
    MODEL_ARGS="${MODEL_ARGS},enable_prefix_caching=False"

    lm_eval \
        --model vllm \
        --model_args "$MODEL_ARGS" \
        --tasks "$TASKS" \
        --batch_size "$BATCH_SIZE" \
        --output_path "$RESULT_DIR" \
        2>&1 | tee "${RESULT_DIR}.log"

    echo "  ✅ Done: $CONFIG_NAME"
}


CONFIG_NAME=${MODEL_BASE}/config.json
evaluate_model "$MODEL_BASE" "$CONFIG_NAME"
