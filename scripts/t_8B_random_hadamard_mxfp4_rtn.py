from auto_round import AutoRound

# Load a model (supports FP8/BF16/FP16/FP32)
model_name_or_path = "Qwen/Qwen3-8B"
output_dir = "./Qwen3-8_autoround_random_hadamard_rtn_mxfp4"

# Ignore layers that should remain in FP16/BF16, matching llm-compressor's ignore list:
#   - visual: all visual encoder layers (not suitable for FP4 quantization)
#   - lm_head: output head (keep high precision for generation quality)
#   - mlp.gate: MoE router (tiny params, precision-sensitive)
#   - linear_attn: fused attention projection (MLA architecture)
#   - shared_expert_gate: shared expert gate (tiny params)
#   - embed_tokens: embedding layer
fp_layers = "visual,lm_head,mlp.gate,linear_attn,shared_expert_gate,embed_tokens,self_attn"

ar = AutoRound(
    model_name_or_path,
    scheme="MXFP4",
    iters=0,
    device_map="auto",
    low_gpu_mem_usage=True,
    #ignore_layers=fp_layers,
    rotation_config="random_hadamard"
)

#ar.quantize_and_save(output_dir=output_dir, format="llm_compressor")
ar.quantize_and_save(output_dir=output_dir, format="auto_round")
