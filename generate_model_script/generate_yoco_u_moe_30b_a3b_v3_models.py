#!/usr/bin/env python3
"""
Generate YOCO-U-MoE-30B-A3B-V3 model (F16 and I2_S).

YOCO-U variant with T=3 self-decoder weight sharing (universal loop).
The self-decoder has N_SELF_LAYERS unique layers that are looped T=3 times,
producing N_SELF_LAYERS*T unrolled self-decoder layers in total, followed
by N_CROSS_LAYERS cross-decoder layers with independent weights.

Architecture:
  d_model=3072, d_ffn=9216, head=32, cross_head=32,
  kv_head=8, cross_kv_head=8, head_dim=128,
  yoco_u_iters=3 (T=3),
  n_self_layers=10 (unique), n_cross_layers=10,
  n_total_layers = 10*3 + 10 = 40 (unrolled),
  yoco_window_size=512, diff_v3=True, weight_tying=True,
  MoE: moe_ffn_dim=1280, moe_expert_num=128, moe_top_k=8,
       d_shared_expert=1280, moe_latent_dim=0,
  Quantisation: bit_linear="attn_qkv,attn_out,attn_gate,yoco_kv,ffn",
                moe_expert_bit_linear=True, act_quant_method="adp_8bits",
                weight_quant_method="TernarySEQ",
  embproj=True, emb_proj_norm=True,
  int8_embedding=True, int8_lm_head=True.

Usage:
  python generate_yoco_u_moe_30b_a3b_v3_models.py [--output-dir DIR]
"""
from __future__ import annotations
import sys
import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "bitnet_backbone-main" / "3rdparty" / "llama.cpp" / "gguf-py"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "BitNet" / "3rdparty" / "llama.cpp" / "gguf-py"))
import gguf

logger = logging.getLogger("generate-yoco-u-moe-30b-a3b-v3")

# ---------- architecture constants ----------
N_SELF_LAYERS = 10       # unique self-decoder layers
N_CROSS_LAYERS = 10      # cross-decoder layers
T_ITERS = 3              # universal loop iterations
N_SELF_UNROLLED = N_SELF_LAYERS * T_ITERS  # 30
N_TOTAL_LAYERS = N_SELF_UNROLLED + N_CROSS_LAYERS  # 40

MOE_EXPERT_NUM = 128
MOE_FFN_DIM = 1280
MOE_TOP_K = 8
D_SHARED_EXPERT = 1280

YOCO_U_MOE_30B_A3B_V3_CONFIG = {
    "hidden_size": 3072,
    "intermediate_size": 9216,
    "num_hidden_layers": N_TOTAL_LAYERS,
    "num_self_layers": N_SELF_LAYERS,
    "num_cross_layers": N_CROSS_LAYERS,
    "yoco_u_iters": T_ITERS,
    "num_attention_heads": 32,
    "num_cross_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_cross_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 154880,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "architectures": ["BitnetForCausalLM"],
    "model_type": "bitnet",
    "torch_dtype": "float32",
    "yoco_window_size": 512,
    "qk_norm": False,
    "qk_rms_clip": True,
    "diff_v3": True,
    "weight_tying": True,
    # MoE
    "moe": True,
    "moe_ffn_dim": MOE_FFN_DIM,
    "moe_expert_num": MOE_EXPERT_NUM,
    "moe_top_k": MOE_TOP_K,
    "d_shared_expert": D_SHARED_EXPERT,
    "moe_latent_dim": 0,
    "moe_latent_norm": False,
    # Quantisation
    "bit_linear": "attn_qkv,attn_out,attn_gate,yoco_kv,ffn",
    "moe_expert_bit_linear": True,
    "act_quant_method": "adp_8bits",
    "weight_quant_method": "TernarySEQ",
    # Embedding
    "embproj": True,
    "emb_proj_norm": True,
    "int8_embedding": True,
    "int8_lm_head": True,
}


def create_hf_model_dir(output_dir: Path, config: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    tokenizer_dst = output_dir / "tokenizer.model"
    if not tokenizer_dst.exists():
        try:
            from huggingface_hub import hf_hub_download
            src = hf_hub_download("1bitLLM/bitnet_b1_58-large", "tokenizer.model")
            import shutil
            shutil.copy(src, tokenizer_dst)
            logger.info(f"Downloaded tokenizer from HuggingFace to {tokenizer_dst}")
        except Exception as e:
            logger.error(f"Failed to download tokenizer from HuggingFace: {e}")
            sys.exit(1)
    return output_dir


def add_vocab(writer, model_dir, vocab_size):
    from sentencepiece import SentencePieceProcessor
    tokenizer_path = model_dir / "tokenizer.model"
    tokenizer = SentencePieceProcessor(str(tokenizer_path))

    tokens, scores, toktypes = [], [], []
    for i in range(tokenizer.vocab_size()):
        tokens.append(tokenizer.id_to_piece(i).encode("utf-8"))
        scores.append(tokenizer.get_score(i))
        if tokenizer.is_unknown(i):
            toktypes.append(gguf.TokenType.UNKNOWN)
        elif tokenizer.is_control(i):
            toktypes.append(gguf.TokenType.CONTROL)
        elif tokenizer.is_unused(i):
            toktypes.append(gguf.TokenType.UNUSED)
        elif tokenizer.is_byte(i):
            toktypes.append(gguf.TokenType.BYTE)
        else:
            toktypes.append(gguf.TokenType.NORMAL)

    while len(tokens) < vocab_size:
        tokens.append(f"[PAD{len(tokens)}]".encode())
        scores.append(-1000.0)
        toktypes.append(gguf.TokenType.UNUSED)

    writer.add_tokenizer_model("llama")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(model_dir, n_vocab=len(tokens))
    special_vocab.add_to_gguf(writer)


def add_model_params(writer, config, name, file_type):
    hidden = config["hidden_size"]
    n_heads = config["num_attention_heads"]
    n_heads_cross = config["num_cross_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    n_cross_kv_heads = config["num_cross_key_value_heads"]
    head_dim = config["head_dim"]
    n_layers = config["num_hidden_layers"]
    n_self = config["num_self_layers"]
    n_cross = config["num_cross_layers"]
    T = config["yoco_u_iters"]
    n_self_unrolled = n_self * T

    writer.add_name(name)
    writer.add_block_count(n_layers)
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(config["intermediate_size"])

    # Per-layer head counts: self-decoder (unrolled) + cross-decoder
    n_head_arr = [n_heads] * n_self_unrolled + [n_heads_cross] * n_cross
    n_head_kv_arr = [n_kv_heads] * n_self_unrolled + [n_cross_kv_heads] * n_cross
    writer.add_head_count(n_head_arr)
    writer.add_head_count_kv(n_head_kv_arr)

    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_rope_freq_base(config["rope_theta"])
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_vocab_size(config["vocab_size"])
    writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
    writer.add_rope_scaling_factor(1.0)
    writer.add_file_type(file_type)
    writer.add_sliding_window(config["yoco_window_size"])

    # YOCO-U iteration count
    writer.add_uint32("bitnet.yoco_u_iters", T)

    # MoE metadata
    writer.add_uint32("bitnet.moe_expert_num", config["moe_expert_num"])
    writer.add_uint32("bitnet.moe_top_k", config["moe_top_k"])
    writer.add_uint32("bitnet.moe_ffn_dim", config["moe_ffn_dim"])
    writer.add_uint32("bitnet.d_shared_expert", config["d_shared_expert"])
    writer.add_uint32("bitnet.moe_latent_dim", config["moe_latent_dim"])


# ---------- tensor shape generators ----------

def generate_self_layer_attn_tensors(config):
    """Self-decoder attention tensors (diff_v3: q has 2x heads for gating)."""
    hidden = config["hidden_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]

    q_heads = n_heads * 2
    q_dim = q_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    tensors = {
        "input_layernorm.weight": (hidden,),
        "self_attn.q_proj.weight": (q_dim, hidden),
        "self_attn.k_proj.weight": (kv_dim, hidden),
        "self_attn.v_proj.weight": (kv_dim, hidden),
        "self_attn.o_proj.weight": (hidden, n_heads * head_dim),
        "self_attn.gate_proj.weight": (2 * n_heads, hidden),
        "self_attn.q_norm.weight": (q_dim,),
        "self_attn.k_norm.weight": (kv_dim,),
        "post_attention_layernorm.weight": (hidden,),
    }
    return tensors


def generate_cross_layer_attn_tensors(config):
    """Cross-decoder attention tensors (diff_v3: q has 2x cross_heads)."""
    hidden = config["hidden_size"]
    n_heads_cross = config["num_cross_attention_heads"]
    head_dim = config["head_dim"]

    q_heads = n_heads_cross * 2
    q_dim = q_heads * head_dim

    tensors = {
        "input_layernorm.weight": (hidden,),
        "self_attn.q_proj.weight": (q_dim, hidden),
        "self_attn.o_proj.weight": (hidden, n_heads_cross * head_dim),
        "self_attn.gate_proj.weight": (2 * n_heads_cross, hidden),
        "self_attn.q_norm.weight": (q_dim,),
        "post_attention_layernorm.weight": (hidden,),
    }
    return tensors


def generate_moe_tensors(config):
    """MoE layer tensors: router, packed expert weights w13/w2, shared expert."""
    hidden = config["hidden_size"]
    n_experts = config["moe_expert_num"]
    ffn_dim = config["moe_ffn_dim"]
    d_shared = config["d_shared_expert"]
    latent_dim = config["moe_latent_dim"]

    expert_input_dim = latent_dim if latent_dim > 0 else hidden

    tensors = {
        "mlp.gate.weight": (n_experts, hidden),
        "mlp.w13": (n_experts, 2 * ffn_dim, expert_input_dim),
        "mlp.w2": (n_experts, expert_input_dim, ffn_dim),
    }

    if latent_dim > 0:
        tensors["mlp.fc1_latent_proj.weight"] = (latent_dim, hidden)
        tensors["mlp.fc2_latent_proj.weight"] = (hidden, latent_dim)
        if config["moe_latent_norm"]:
            tensors["mlp.fc1_latent_norm.weight"] = (latent_dim,)
            tensors["mlp.fc2_latent_norm.weight"] = (latent_dim,)

    if d_shared > 0:
        tensors["mlp.shared.gate_proj.weight"] = (d_shared, hidden)
        tensors["mlp.shared.up_proj.weight"] = (d_shared, hidden)
        tensors["mlp.shared.down_proj.weight"] = (hidden, d_shared)
        tensors["mlp.shared_gate.weight"] = (1, hidden)

    return tensors


def generate_shared_cross_kv_tensors(config):
    """Shared YOCO cross KV projection tensors."""
    hidden = config["hidden_size"]
    n_cross_kv_heads = config["num_cross_key_value_heads"]
    head_dim = config["head_dim"]
    kv_dim = n_cross_kv_heads * head_dim

    return {
        "yoco_norm.weight": (hidden,),
        "k_proj.weight": (kv_dim, hidden),
        "v_proj.weight": (kv_dim, hidden),
        "k_norm.weight": (kv_dim,),
    }


def generate_embproj_tensors(config):
    """Embedding projection tensors (embproj=True)."""
    hidden = config["hidden_size"]
    return {
        "emb_proj_in.weight": (hidden, hidden),
        "emb_proj_out.weight": (hidden, hidden),
        "emb_in_norm.weight": (hidden,),
        "emb_out_norm.weight": (hidden,),
    }


def weight_quant_ternary(weight: np.ndarray) -> np.ndarray:
    w = weight.astype(np.float32)
    s = 1.0 / max(np.abs(w).mean(), 1e-5)
    return np.clip(np.round(w * s), -1, 1) / s


def _write_all_tensors(writer, config, tensor_map, dtype=np.float32):
    """Write all model tensors with T=3 self-decoder weight sharing.

    Self-decoder: 10 unique layers, each shared across T=3 iterations.
    GGUF contains 30 unrolled self-decoder layers (0..29) where layers
    i and i+10 and i+20 share the same weight data (copied).
    Cross-decoder: 10 independent layers (30..39).
    """
    hidden = config["hidden_size"]
    n_self = config["num_self_layers"]
    n_cross = config["num_cross_layers"]
    T = config["yoco_u_iters"]
    n_self_unrolled = n_self * T
    vocab = config["vocab_size"]

    quant_names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    def _make_tensor(shape):
        return np.random.randn(*shape).astype(np.float32)

    def _finalize(data):
        return data.astype(dtype)

    # --- Embedding ---
    data = _finalize(_make_tensor((vocab, hidden)))
    name = tensor_map.get_name("model.embed_tokens.weight", try_suffixes=(".weight",))
    writer.add_tensor(name, data)

    # --- Self-decoder: generate 10 unique weight sets, copy across T=3 iterations ---
    self_attn_template = generate_self_layer_attn_tensors(config)
    moe_template = generate_moe_tensors(config)

    # Pre-generate unique weights for each of the 10 self-decoder layers
    self_attn_weights = {}
    for suffix, shape in self_attn_template.items():
        self_attn_weights[suffix] = [None] * n_self
    moe_weights = {}
    for suffix, shape in moe_template.items():
        moe_weights[suffix] = [None] * n_self

    for sl in range(n_self):
        for suffix, shape in self_attn_template.items():
            data = _make_tensor(shape)
            should_quant = any(qn in suffix for qn in quant_names)
            if should_quant and len(shape) == 2:
                data = weight_quant_ternary(data)
            self_attn_weights[suffix][sl] = _finalize(data)

        for suffix, shape in moe_template.items():
            data = _make_tensor(shape)
            should_quant = any(qn in suffix for qn in quant_names)
            if should_quant and len(shape) == 2:
                data = weight_quant_ternary(data)
            moe_weights[suffix][sl] = _finalize(data)

    # Write unrolled self-decoder layers: T iterations x N_SELF_LAYERS
    for t in range(T):
        for sl in range(n_self):
            i = t * n_self + sl  # unrolled layer index (0..29)

            # Attention (shared weights)
            for suffix in self_attn_template:
                tensor_name = f"model.layers.{i}.{suffix}"
                data = self_attn_weights[suffix][sl].copy()
                mapped = tensor_map.get_name(tensor_name, try_suffixes=(".weight",))
                if mapped is None:
                    mapped = tensor_name.replace("model.", "")
                writer.add_tensor(mapped, data)

            # MoE FFN (shared weights)
            for suffix in moe_template:
                tensor_name = f"model.layers.{i}.{suffix}"
                data = moe_weights[suffix][sl].copy()
                mapped = tensor_map.get_name(tensor_name, try_suffixes=(".weight",))
                if mapped is None:
                    mapped = tensor_name.replace("model.", "")
                writer.add_tensor(mapped, data)

    # --- Cross-decoder layers (n_self_unrolled .. n_total-1) ---
    cross_attn_template = generate_cross_layer_attn_tensors(config)

    for cl in range(n_cross):
        i = n_self_unrolled + cl  # layer index (30..39)

        # Cross attention (no K/V proj - uses shared YOCO KV)
        for suffix, shape in cross_attn_template.items():
            tensor_name = f"model.layers.{i}.{suffix}"
            data = _make_tensor(shape)
            should_quant = any(qn in suffix for qn in quant_names)
            if should_quant and len(shape) == 2:
                data = weight_quant_ternary(data)
            mapped = tensor_map.get_name(tensor_name, try_suffixes=(".weight",))
            if mapped is None:
                mapped = tensor_name.replace("model.", "")
            writer.add_tensor(mapped, _finalize(data))

        # MoE FFN (independent per cross layer)
        for suffix, shape in moe_template.items():
            tensor_name = f"model.layers.{i}.{suffix}"
            data = _make_tensor(shape)
            should_quant = any(qn in suffix for qn in quant_names)
            if should_quant and len(shape) == 2:
                data = weight_quant_ternary(data)
            mapped = tensor_map.get_name(tensor_name, try_suffixes=(".weight",))
            if mapped is None:
                mapped = tensor_name.replace("model.", "")
            writer.add_tensor(mapped, _finalize(data))

    # --- Shared YOCO cross KV tensors ---
    shared_tensors = generate_shared_cross_kv_tensors(config)
    for tname, shape in shared_tensors.items():
        data = _make_tensor(shape)
        if len(shape) == 2:
            data = weight_quant_ternary(data)
        writer.add_tensor(tname, _finalize(data))

    # --- Embedding projections ---
    if config.get("embproj", False):
        embproj_tensors = generate_embproj_tensors(config)
        for tname, shape in embproj_tensors.items():
            writer.add_tensor(tname, _finalize(_make_tensor(shape)))

    # --- Final norm ---
    data = _finalize(_make_tensor((hidden,)))
    name = tensor_map.get_name("model.norm.weight", try_suffixes=(".weight",))
    writer.add_tensor(name, data)


def generate_f16_gguf(model_dir: Path, output_path: Path, config: dict):
    """Generate a F16 GGUF model."""
    logger.info("Generating YOCO-U-MoE-30B-A3B-V3 F16 GGUF")

    n_layers = config["num_hidden_layers"]

    writer = gguf.GGUFWriter(output_path, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.BITNET])
    add_model_params(writer, config, "yoco-u-moe-30b-a3b-v3-bitnet", gguf.GGMLQuantizationType.F16)
    add_vocab(writer, model_dir, config["vocab_size"])

    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.BITNET, n_layers)
    _write_all_tensors(writer, config, tensor_map, dtype=np.float16)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    f16_size = output_path.stat().st_size / (1024**3)
    logger.info(f"  F16 model saved: {output_path} ({f16_size:.2f} GB)")


def generate_i2s_gguf(f16_path: Path, output_path: Path):
    """Quantize an existing F16 GGUF to I2_S."""
    logger.info(f"Quantizing F16 -> I2_S (with embedding Q8_0)")
    logger.info(f"  Source: {f16_path}")

    quantize_bin = Path("/home/azureuser/huangxin/code_list/bitnet-moe-script/build_script/build_bin_yoco_moe/bin/llama-quantize")
    if not quantize_bin.exists():
        quantize_bin = Path("/home/azureuser/huangxin/code_list/bitnet-yoco-u-script/build_script/build_bin_yoco_u/bin/llama-quantize")
    if not quantize_bin.exists():
        logger.error(f"llama-quantize not found at {quantize_bin}")
        sys.exit(1)

    import subprocess
    cmd = [
        str(quantize_bin),
        "--token-embedding-type", "Q8_0",
        str(f16_path),
        str(output_path),
        "I2_S",
        "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Quantization failed:\n{result.stderr}\n{result.stdout}")
        sys.exit(1)

    f16_size = f16_path.stat().st_size / (1024**3)
    i2s_size = output_path.stat().st_size / (1024**2)
    logger.info(f"  I2_S model saved: {output_path} ({i2s_size:.1f} MB)")
    logger.info(f"  Compression ratio: {f16_size*1024/i2s_size:.1f}x")


def _count_params(shapes):
    return sum(int(np.prod(s)) for s in shapes.values())


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOCO-U-MoE-30B-A3B-V3 model with T=3 weight sharing (F16 and I2_S)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="/home/azureuser/models/yoco-moe-models/yoco-u-moe-30b-a3b-v3",
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = YOCO_U_MOE_30B_A3B_V3_CONFIG
    n_self = config["num_self_layers"]
    n_cross = config["num_cross_layers"]
    T = config["yoco_u_iters"]
    n_self_unrolled = n_self * T
    n_layers = config["num_hidden_layers"]

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Create HF dir (for tokenizer)
    hf_dir = output_base / "yoco-u-moe-30b-a3b-v3-hf"
    create_hf_model_dir(hf_dir, config)

    # Generate F16 model
    f16_dir = output_base / "yoco-u-moe-30b-a3b-v3-bitnet-f16"
    f16_dir.mkdir(parents=True, exist_ok=True)
    f16_path = f16_dir / "ggml-model-f16.gguf"
    generate_f16_gguf(hf_dir, f16_path, config)

    # Generate I2_S model (quantized from F16)
    i2s_dir = output_base / "yoco-u-moe-30b-a3b-v3-bitnet-i2s"
    i2s_dir.mkdir(parents=True, exist_ok=True)
    i2s_path = i2s_dir / "ggml-model-i2_s.gguf"
    generate_i2s_gguf(f16_path, i2s_path)

    # Count params
    self_attn_params = _count_params(generate_self_layer_attn_tensors(config))
    cross_attn_params = _count_params(generate_cross_layer_attn_tensors(config))
    moe_params = _count_params(generate_moe_tensors(config))
    shared_kv_params = _count_params(generate_shared_cross_kv_tensors(config))
    embproj_params = _count_params(generate_embproj_tensors(config)) if config.get("embproj") else 0
    emb_params = config["vocab_size"] * config["hidden_size"]
    norm_params = config["hidden_size"]

    # Unique params: self-decoder layers counted once (not T times)
    unique_params = (
        (self_attn_params + moe_params) * n_self
        + (cross_attn_params + moe_params) * n_cross
        + shared_kv_params + embproj_params + emb_params + norm_params
    )

    # Compute params: self-decoder layers counted T times (unrolled)
    compute_params = (
        (self_attn_params + moe_params) * n_self * T
        + (cross_attn_params + moe_params) * n_cross
        + shared_kv_params + embproj_params + emb_params + norm_params
    )

    print("\n" + "=" * 70)
    print("YOCO-U-MoE-30B-A3B-V3 Models Generated (F16 + I2_S)!")
    print(f"  F16 Model:  {f16_path}")
    print(f"  F16 Size:   {f16_path.stat().st_size / (1024**3):.2f} GB")
    print(f"  I2_S Model: {i2s_path}")
    print(f"  I2_S Size:  {i2s_path.stat().st_size / (1024**2):.1f} MB")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  d_model={config['hidden_size']}, d_ffn={config['intermediate_size']}")
    print(f"  head={config['num_attention_heads']}, cross_head={config['num_cross_attention_heads']}")
    print(f"  kv_head={config['num_key_value_heads']}, cross_kv_head={config['num_cross_key_value_heads']}")
    print(f"  head_dim={config['head_dim']}")
    print(f"  n_total_layers={n_layers} ({n_self} self x T={T} + {n_cross} cross)")
    print(f"  yoco_u_iters={T} (universal loop)")
    print(f"  yoco_window_size={config['yoco_window_size']}")
    print(f"  diff_v3={config['diff_v3']}, weight_tying={config['weight_tying']}")
    print(f"\nMoE Config:")
    print(f"  moe_expert_num={config['moe_expert_num']}, moe_top_k={config['moe_top_k']}")
    print(f"  moe_ffn_dim={config['moe_ffn_dim']}, d_shared_expert={config['d_shared_expert']}")
    print(f"  moe_latent_dim={config['moe_latent_dim']}")
    print(f"  moe_expert_bit_linear={config['moe_expert_bit_linear']}")
    print(f"\nQuantisation:")
    print(f"  bit_linear={config['bit_linear']}")
    print(f"  act_quant_method={config['act_quant_method']}")
    print(f"  weight_quant_method={config['weight_quant_method']}")
    print(f"  int8_embedding={config['int8_embedding']}, int8_lm_head={config['int8_lm_head']}")
    print(f"  embproj={config['embproj']}, emb_proj_norm={config['emb_proj_norm']}")
    print(f"\nParams:")
    print(f"  Unique (storage):  {unique_params:,} ({unique_params/1e9:.2f}B)")
    print(f"  Compute (per tok): {compute_params:,} ({compute_params/1e9:.2f}B)")
    print(f"  Self-decoder: {n_self} unique layers x T={T} = {n_self_unrolled} unrolled layers (shared weights)")
    print(f"  Cross-decoder: {n_cross} independent layers")


if __name__ == "__main__":
    main()
