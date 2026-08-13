"""Standalone PyTorch inference for merged YOCO Quant-MoE checkpoints.

This file is deliberately self-contained.  It imports only Python's standard
library and PyTorch: it does not import NNScaler, Triton, FlashAttention,
TransformerEngine, DeepGEMM, or any module from this repository.  It is a
single-GPU, batch-one reference/integration implementation for infra teams.

Supported checkpoint format
---------------------------
``metadata.json`` and ``model_state_rank_0.pth`` produced by
``scripts/ckpt_merge.py``.  The runner accepts bare model keys and the common
``model.`` / ``backbone.model.`` prefixes.

Quantized routed experts use checkpoint-native ADP8 + TernarySEQ.  Ternary
weights are materialized once while loading and retained as BF16 inference
weights, so decode does not re-run fake quantization.  Optional blockwise INT8
embedding and LM-head weights are handled in the same way.

This is intentionally a native-PyTorch reference, not a production kernel.
It uses ``torch.nn.functional.scaled_dot_product_attention`` and ordinary
``F.linear`` calls.  The tokenizer is intentionally outside this file: callers
pass token IDs, which makes the file usable by an inference stack with its own
tokenizer and request scheduler.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


MODEL_STATE_NAME = "model_state_rank_0.pth"
METADATA_NAME = "metadata.json"
GROUP_SIZE = 128
TERNARY_CLIP_RATIO = 1.0 - 1e-2
TERNARY_LEVELS = 1.5
TERNARY_EPS = 1e-5


def _parse_bit_linear(value: str | Iterable[str] | None) -> frozenset[str]:
    valid = frozenset(("attn_qkv", "attn_out", "attn_gate", "yoco_kv", "ffn"))
    if value is None:
        return frozenset()
    if isinstance(value, str):
        names = [name.strip().lower() for name in value.split(",") if name.strip()]
    else:
        names = [str(name).strip().lower() for name in value]
    if not names or names == ["off"]:
        return frozenset()
    if names == ["all"]:
        return valid
    unknown = set(names).difference(valid)
    if unknown:
        raise ValueError(f"unknown bit_linear target(s): {sorted(unknown)}")
    return frozenset(names)


@dataclass
class ModelConfig:
    """Architecture fields needed by the standalone YOCO decoder."""

    d_model: int = 3072
    d_ffn: int = 9216
    head: int = 32
    cross_head: int | None = None
    kv_head: int = 8
    cross_kv_head: int = 8
    head_dim: int | None = None
    n_layers: int = 20
    vocab_size: int = 154880
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    qk_norm: bool = False
    qk_rms_clip: bool = True
    qk_rms_gamma: bool = True
    qk_rms_limit: float = 3.0
    attention_bias: bool = False
    weight_tying: bool = False
    headwise_glu: bool = False
    diff_v2: bool = False
    diff_v3: bool = True
    yoco_cross_layers: int = 10
    yoco_window_size: int = 512
    universal_loop: int = 1
    mtp_n: int = 0
    moe: bool = True
    moe_ffn_dim: int = 3840
    moe_expert_num: int = 128
    moe_top_k: int = 8
    d_shared_expert: int = 1280
    dense_layers: int = 0
    moe_latent_dim: int = 1024
    moe_latent_norm: bool = True
    quant_mode: str = "bfloat16"
    quant_block_size: int = 128
    bit_linear: str = "off"
    attn_bit_linear_level: int = 0
    ffn_bit_linear_level: int = 0
    moe_expert_bit_linear: bool = False
    moe_expert_quant_dtp: bool = False
    act_quant_method: str = "per-token"
    moe_intermediate_act_quant_method: str | None = None
    weight_quant_method: str = "TernarySEQ"
    weight_clip_val_init_method: str = "default"
    int8_embedding: bool = False
    int8_embedding_block_size: int = 128
    int8_lm_head: bool = False
    int8_lm_head_block_size: int = 128
    embproj: bool = False
    emb_proj_norm: bool = False
    embedding_dim: int | None = None
    swiglu_limit: float = 10.0

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelConfig":
        valid = {field.name for field in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in valid})

    def validate(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.d_model // self.head
        if self.cross_head is None:
            self.cross_head = self.head
        if self.embedding_dim is None:
            self.embedding_dim = self.d_model
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        # Differential attention can use a wider attention value space than
        # d_model. YOCO-MoE-30B-A3B uses 32 * 128 attention channels with a
        # 3072-wide residual stream, so do not require these widths to match.
        if self.cross_head <= 0 or self.kv_head <= 0 or self.cross_kv_head <= 0:
            raise ValueError("attention head counts must be positive")
        if self.head % self.kv_head:
            raise ValueError("head must be divisible by kv_head")
        if self.cross_head % self.cross_kv_head:
            raise ValueError("cross_head must be divisible by cross_kv_head")
        if self.diff_v2 and self.diff_v3:
            raise ValueError("diff_v2 and diff_v3 are mutually exclusive")
        targets = self.bit_linear_targets
        if "yoco_kv" in targets and not self.yoco_cross_layers:
            raise ValueError("bit_linear includes 'yoco_kv' but the model has no YOCO cross layers")
        if (
            self.bit_linear != "all"
            and "attn_gate" in targets
            and not (self.diff_v2 or self.diff_v3 or self.headwise_glu)
        ):
            raise ValueError("bit_linear includes 'attn_gate' but the model has no attention gate")
        if self.yoco_cross_layers < 0 or self.yoco_cross_layers > self.n_layers:
            raise ValueError("yoco_cross_layers must be in [0, n_layers]")
        if self.mtp_n < 0:
            raise ValueError("mtp_n must be non-negative")
        if self.mtp_n and not self.yoco_cross_layers:
            raise ValueError("MTP requires yoco_cross_layers > 0")
        if self.moe and not 0 < self.moe_top_k <= self.moe_expert_num:
            raise ValueError("moe_top_k must be in [1, moe_expert_num]")
        if self.embedding_dim != self.d_model and not self.embproj:
            raise ValueError("embedding_dim != d_model requires embproj=True")
        if self.moe_expert_bit_linear:
            expert_dim = self.moe_latent_dim or self.d_model
            if expert_dim % GROUP_SIZE or self.moe_ffn_dim % GROUP_SIZE:
                raise ValueError(
                    "ADP8 Quant-MoE requires expert input and moe_ffn_dim "
                    f"to be divisible by {GROUP_SIZE}"
                )
            if self.weight_quant_method.lower() != "ternaryseq":
                raise ValueError("standalone Quant-MoE supports TernarySEQ only")

    @property
    def bit_linear_targets(self) -> frozenset[str]:
        targets = _parse_bit_linear(self.bit_linear)
        if self.attn_bit_linear_level:
            legacy = ("attn_qkv", "attn_out", "yoco_kv")
            targets = targets.union(
                name for bit, name in enumerate(legacy) if self.attn_bit_linear_level & (1 << bit)
            )
        if self.ffn_bit_linear_level & 1:
            targets = targets.union(("ffn",))
        return frozenset(targets)


YOCO_MOE_30B_A3B_V3 = ModelConfig(
    d_model=3072,
    d_ffn=9216,
    head=32,
    cross_head=32,
    kv_head=8,
    cross_kv_head=8,
    head_dim=128,
    n_layers=20,
    qk_norm=False,
    qk_rms_clip=True,
    diff_v3=True,
    weight_tying=True,
    yoco_cross_layers=10,
    yoco_window_size=512,
    moe=True,
    moe_ffn_dim=1280,
    moe_expert_num=128,
    moe_top_k=8,
    d_shared_expert=1280,
    moe_latent_dim=0,
    moe_latent_norm=False,
    bit_linear="attn_qkv,attn_out,attn_gate,yoco_kv,ffn",
    moe_expert_bit_linear=True,
    act_quant_method="adp_8bits",
    weight_quant_method="TernarySEQ",
    embproj=True,
    emb_proj_norm=True,
    int8_embedding=True,
    int8_lm_head=True,
)
YOCO_MOE_30B_A3B_L3 = ModelConfig(
    d_model=3072,
    d_ffn=9216,
    head=32,
    cross_head=32,
    kv_head=8,
    cross_kv_head=8,
    head_dim=128,
    n_layers=20,
    qk_norm=False,
    qk_rms_clip=True,
    diff_v3=True,
    weight_tying=False,
    yoco_cross_layers=10,
    yoco_window_size=512,
    moe=True,
    moe_ffn_dim=3840,
    moe_expert_num=128,
    moe_top_k=8,
    d_shared_expert=1280,
    moe_latent_dim=1024,
    moe_latent_norm=True,
)
MODEL_PRESETS: dict[str, ModelConfig] = {
    "YOCO-MoE-30B-A3B-v3": YOCO_MOE_30B_A3B_V3,
    "YOCO-MoE-30B-A3B-L3": YOCO_MOE_30B_A3B_L3,
}


def _strip_model_prefix(name: str) -> str:
    """Accept merged, trainer-wrapper, and bare model checkpoint keys."""
    prefixes = ("backbone.model.", "backbone.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
                break
    return name


def load_model_args(
    checkpoint_dir: Path,
    fallback_model: str | None = None,
) -> tuple[ModelConfig, dict[str, Any]]:
    """Read architecture metadata without importing the training config module."""
    metadata_path = checkpoint_dir / METADATA_NAME
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        values = metadata.get("modelargs")
        if not isinstance(values, Mapping):
            raise ValueError(f"{metadata_path} has no object-valued 'modelargs' entry")
        return ModelConfig.from_mapping(values), metadata
    if fallback_model is None:
        raise FileNotFoundError(
            f"{metadata_path} is required unless --model selects a built-in preset"
        )
    try:
        return copy.deepcopy(MODEL_PRESETS[fallback_model]), {}
    except KeyError as error:
        raise ValueError(
            f"unknown model preset {fallback_model!r}; choices: {', '.join(MODEL_PRESETS)}"
        ) from error


def apply_inference_quant_overrides(
    model_args: ModelConfig,
    *,
    quant_moe: bool | None,
    int8_embedding: bool | None,
    int8_lm_head: bool | None,
    mtp_n: int | None = None,
) -> None:
    """Apply explicit runtime choices to metadata-derived model arguments."""
    if quant_moe is not None:
        model_args.moe_expert_bit_linear = quant_moe
        if quant_moe:
            model_args.quant_mode = "bfloat16"
            model_args.act_quant_method = "adp_8bits"
            model_args.moe_intermediate_act_quant_method = "adp_8bits"
            model_args.weight_quant_method = "TernarySEQ"
    if int8_embedding is not None:
        model_args.int8_embedding = int8_embedding
    if int8_lm_head is not None:
        model_args.int8_lm_head = int8_lm_head
    if mtp_n is not None:
        model_args.mtp_n = mtp_n
    model_args.validate()


def load_model_state(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Load a trusted merged state dict, normalizing only leading wrappers."""
    path = checkpoint_dir / MODEL_STATE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"missing merged model state: {path}")
    try:
        loaded = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except (TypeError, RuntimeError):
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, Mapping) and isinstance(loaded.get("model"), Mapping):
        loaded = loaded["model"]
    if not isinstance(loaded, Mapping):
        raise TypeError(f"{path} must contain a state-dict mapping")
    state = {
        _strip_model_prefix(str(name)): tensor
        for name, tensor in loaded.items()
        if isinstance(tensor, torch.Tensor)
    }
    if not any(name.startswith("layers.0.") for name in state):
        raise ValueError(
            "expected a merged Model state dict with keys such as "
            "'layers.0.self_attn.q_proj.weight'; sharded NNScaler checkpoints "
            "must be merged with scripts/ckpt_merge.py first"
        )
    return state


def _groupwise_int8_fake_quant(x: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """The inference value of the repository's per-group fake INT8 quantizer."""
    if x.shape[-1] % group_size:
        raise ValueError(
            f"last dimension {x.shape[-1]} is not divisible by group size {group_size}"
        )
    dtype = x.dtype
    grouped = x.float().reshape(*x.shape[:-1], -1, group_size)
    scale = 127.0 / grouped.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-5)
    return ((grouped * scale).round().clamp(-128, 127) / scale).reshape_as(x).to(dtype)


def _adp8_quant(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # The affine is intentionally FP32, but the training implementation casts
    # the fake-quantized value back to the original activation dtype.
    return _groupwise_int8_fake_quant(x.float() * scale + bias).to(x.dtype)


def _blockwise_int8_weight(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    """Materialize the inference value of WeightQuantBlockwiseINT8."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    source = weight.float()
    width = source.shape[-1]
    padded_width = math.ceil(width / block_size) * block_size
    if padded_width != width:
        source = F.pad(source, (0, padded_width - width))
    grouped = source.reshape(*source.shape[:-1], -1, block_size)
    scale = 127.0 / grouped.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-5)
    quantized = ((grouped * scale).round().clamp(-127, 127) / scale).reshape(
        *source.shape[:-1], padded_width
    )
    return quantized[..., :width].to(torch.bfloat16)


def _ternaryseq_weight(weight: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    """Materialize Q = alpha * round(clamp(W / alpha) * 1.5) / 1.5."""
    if tuple(clip.shape) != tuple(weight.shape[:-1]):
        raise ValueError(
            "TernarySEQ clip shape must equal the weight shape without its input dimension: "
            f"weight={tuple(weight.shape)}, clip={tuple(clip.shape)}"
        )
    alpha = clip.float().clamp_min(TERNARY_EPS).unsqueeze(-1)
    quantized = weight.float()
    quantized.div_(alpha)
    quantized.clamp_(-TERNARY_CLIP_RATIO, TERNARY_CLIP_RATIO)
    quantized.mul_(TERNARY_LEVELS).round_().div_(TERNARY_LEVELS).mul_(alpha)
    return quantized.to(torch.bfloat16)


class _TensorStore:
    """Moves one tensor at a time so checkpoint loading has bounded peak CPU state."""

    def __init__(self, state: Mapping[str, torch.Tensor], device: torch.device):
        self._state = dict(state)
        self.device = device

    def has(self, name: str) -> bool:
        return name in self._state

    def take(
        self,
        name: str,
        *,
        dtype: torch.dtype | None = None,
        required: bool = True,
    ) -> torch.Tensor | None:
        value = self._state.pop(name, None)
        if value is None:
            if required:
                raise KeyError(f"checkpoint is missing required tensor {name!r}")
            return None
        if dtype is None:
            return value.to(device=self.device)
        return value.to(device=self.device, dtype=dtype)

    def take_any(
        self,
        names: Sequence[str],
        *,
        dtype: torch.dtype | None = None,
        required: bool = True,
    ) -> torch.Tensor | None:
        for name in names:
            if self.has(name):
                return self.take(name, dtype=dtype)
        if required:
            raise KeyError(f"checkpoint is missing all aliases: {', '.join(names)}")
        return None

    def clear(self) -> None:
        self._state.clear()


@dataclass
class _Linear:
    weight: torch.Tensor
    bias: torch.Tensor | None = None
    act_quant_method: str | None = None
    act_scale: torch.Tensor | None = None
    act_bias: torch.Tensor | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
        if self.act_quant_method == "adp_8bits":
            assert self.act_scale is not None and self.act_bias is not None
            flat = _adp8_quant(flat, self.act_scale, self.act_bias)
        elif self.act_quant_method == "per-token":
            scale = 127.0 / flat.float().abs().amax(dim=-1, keepdim=True).clamp_min_(1e-5)
            flat = ((flat.float() * scale).round().clamp(-128, 127) / scale).to(flat.dtype)
        elif self.act_quant_method == "per-group":
            flat = _groupwise_int8_fake_quant(flat)
        elif self.act_quant_method is not None:
            raise ValueError(f"unsupported activation quantization method {self.act_quant_method!r}")
        return F.linear(flat, self.weight, self.bias).view(*x.shape[:-1], self.weight.shape[0])


@dataclass
class _HeadLinear:
    linear: _Linear
    num_heads: int
    head_dim: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear(x)
        return output.view(*x.shape[:-1], self.num_heads, self.head_dim)


@dataclass
class _RMSNorm:
    weight: torch.Tensor | None
    eps: float

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x.to(torch.bfloat16),
            (x.shape[-1],),
            weight=None if self.weight is None else self.weight.to(torch.bfloat16),
            eps=self.eps,
        )


@dataclass
class _RMSClip:
    weight: torch.Tensor | None
    eps: float
    limit: float

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        coefficient = (
            self.limit * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        ).clamp(max=1.0)
        result = (x_float * coefficient).to(x.dtype)
        return result if self.weight is None else result * self.weight.to(result.dtype)


class _Identity:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x


@dataclass
class _Router:
    normalized_weight: torch.Tensor

    @classmethod
    def from_weight(cls, weight: torch.Tensor) -> "_Router":
        normalized = weight.float()
        normalized.div_(normalized.norm(dim=1, keepdim=True).clamp_min_(1e-6))
        return cls(normalized)

    def route(self, x: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = torch.softmax(F.linear(x.float(), self.normalized_weight), dim=-1)
        scores, experts = torch.topk(probabilities, top_k, dim=-1)
        return scores / scores.sum(dim=-1, keepdim=True), experts


@dataclass
class _FeedForward:
    up_proj: _Linear
    gate_proj: _Linear
    down_proj: _Linear
    swiglu_limit: float

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        hidden = F.silu(gate.clamp(max=self.swiglu_limit)) * up.clamp(
            min=-self.swiglu_limit, max=self.swiglu_limit
        )
        return self.down_proj(hidden)


@dataclass
class _MoE:
    router: _Router
    w13: torch.Tensor
    w2: torch.Tensor
    top_k: int
    swiglu_limit: float
    quantized: bool
    w13_act_scale: torch.Tensor | None
    w13_act_bias: torch.Tensor | None
    w2_act_scale: torch.Tensor | None
    w2_act_bias: torch.Tensor | None
    fc1_latent_proj: _Linear | None
    fc2_latent_proj: _Linear | None
    fc1_latent_norm: _RMSNorm | _Identity
    fc2_latent_norm: _RMSNorm | _Identity
    shared: _FeedForward | None
    shared_gate: _Linear | None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scores, experts = self.router.route(x, self.top_k)
        expert_input = x
        if self.fc1_latent_proj is not None:
            expert_input = self.fc1_latent_norm(self.fc1_latent_proj(expert_input))

        output = torch.zeros_like(expert_input, dtype=torch.bfloat16)
        for expert in range(self.w13.shape[0]):
            token_slot = (experts == expert).nonzero(as_tuple=False)
            if token_slot.numel() == 0:
                continue
            token_ids = token_slot[:, 0]
            slots = token_slot[:, 1]
            hidden = expert_input.index_select(0, token_ids).to(torch.bfloat16)
            if self.quantized:
                assert self.w13_act_scale is not None and self.w13_act_bias is not None
                assert self.w2_act_scale is not None and self.w2_act_bias is not None
                hidden = _adp8_quant(
                    hidden, self.w13_act_scale[expert], self.w13_act_bias[expert]
                )
            w13_output = F.linear(hidden, self.w13[expert])
            gate, up = w13_output.chunk(2, dim=-1)
            intermediate = (
                F.silu(gate.float().clamp(max=self.swiglu_limit))
                * up.float().clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
                * scores[token_ids, slots].float().unsqueeze(-1)
            ).to(torch.bfloat16)
            if self.quantized:
                intermediate = _adp8_quant(
                    intermediate, self.w2_act_scale[expert], self.w2_act_bias[expert]
                )
            expert_output = F.linear(intermediate, self.w2[expert])
            output.index_add_(0, token_ids, expert_output)

        if self.fc2_latent_proj is not None:
            output = self.fc2_latent_proj(self.fc2_latent_norm(output))
        if self.shared is not None:
            assert self.shared_gate is not None
            output = output + torch.sigmoid(self.shared_gate(x)) * self.shared(x)
        return output


class _KVCache:
    def __init__(
        self,
        *,
        self_layers: int,
        max_seq_len: int,
        kv_head: int,
        cross_kv_head: int,
        head_dim: int,
        has_yoco_cache: bool,
        device: torch.device,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.length = 0
        self.self_key = [
            torch.empty((max_seq_len, kv_head, head_dim), dtype=torch.bfloat16, device=device)
            for _ in range(self_layers)
        ]
        self.self_value = [torch.empty_like(value) for value in self.self_key]
        self.yoco_key = (
            torch.empty((max_seq_len, cross_kv_head, head_dim), dtype=torch.bfloat16, device=device)
            if has_yoco_cache
            else None
        )
        self.yoco_value = torch.empty_like(self.yoco_key) if self.yoco_key is not None else None

    def reset(self) -> None:
        self.length = 0

    def append_self(self, slot: int, start: int, key: torch.Tensor, value: torch.Tensor) -> None:
        end = start + key.shape[0]
        self.self_key[slot][start:end].copy_(key)
        self.self_value[slot][start:end].copy_(value)

    def append_yoco(self, start: int, key: torch.Tensor, value: torch.Tensor) -> None:
        assert self.yoco_key is not None and self.yoco_value is not None
        end = start + key.shape[0]
        self.yoco_key[start:end].copy_(key)
        self.yoco_value[start:end].copy_(value)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.outer(positions.float(), inv_freq)
    cos, sin = freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        first, second = x.float().chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(x.dtype)

    return rotate(query), rotate(key)


def _native_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    window_size: int,
) -> torch.Tensor:
    """Causal GQA attention using only native PyTorch SDPA."""
    q_heads = query.shape[1]
    kv_heads = key.shape[1]
    if q_heads % kv_heads:
        raise ValueError(f"query heads ({q_heads}) must divide KV heads ({kv_heads})")
    if q_heads != kv_heads:
        repeat = q_heads // kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    key_positions = torch.arange(key.shape[0], device=query.device)
    mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    if window_size > 0:
        mask &= key_positions.unsqueeze(0) >= (query_positions - window_size).unsqueeze(1)
    query_4d = query.transpose(0, 1).unsqueeze(0)
    key_4d = key.transpose(0, 1).unsqueeze(0)
    value_4d = value.transpose(0, 1).unsqueeze(0)
    output = F.scaled_dot_product_attention(
        query_4d,
        key_4d,
        value_4d,
        attn_mask=mask.unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
    )
    return output.squeeze(0).transpose(0, 1)


@dataclass
class _Attention:
    is_self_layer: bool
    q_proj: _HeadLinear
    k_proj: _HeadLinear | None
    v_proj: _HeadLinear | None
    q_norm: _RMSNorm | _RMSClip | _Identity
    k_norm: _RMSNorm | _RMSClip | _Identity
    o_proj: _Linear
    gate_proj: _Linear | None
    diff_v2: bool
    diff_v3: bool
    headwise_glu: bool
    window_size: int
    inv_freq: torch.Tensor | None

    def __call__(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: _KVCache,
        *,
        cache_slot: int | None,
        cache_start: int,
        yoco_key: torch.Tensor | None = None,
        yoco_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.q_norm(self.q_proj(x))
        if self.is_self_layer:
            assert self.k_proj is not None and self.v_proj is not None
            assert self.inv_freq is not None and cache_slot is not None
            key = self.k_norm(self.k_proj(x))
            value = self.v_proj(x)
            query, key = _apply_rotary(query, key, positions, self.inv_freq)
            cache.append_self(cache_slot, cache_start, key, value)
            end = cache_start + x.shape[0]
            key = cache.self_key[cache_slot][:end]
            value = cache.self_value[cache_slot][:end]
        else:
            if yoco_key is None or yoco_value is None:
                raise RuntimeError("cross decoder attention requires YOCO key/value cache")
            end = cache_start + x.shape[0]
            key, value = yoco_key[:end], yoco_value[:end]

        output = _native_attention(
            query,
            key,
            value,
            positions,
            window_size=self.window_size,
        )
        if self.diff_v3:
            assert self.gate_proj is not None
            output = output * torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
            output = output[:, 0::2] - output[:, 1::2]
        elif self.diff_v2:
            assert self.gate_proj is not None
            first, second = output[:, 0::2], output[:, 1::2]
            output = first - torch.sigmoid(self.gate_proj(x)).unsqueeze(-1) * second
        elif self.headwise_glu:
            assert self.gate_proj is not None
            output = output * torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        return self.o_proj(output.reshape(output.shape[0], -1))


@dataclass
class _Block:
    input_norm: _RMSNorm
    post_attention_norm: _RMSNorm
    attention: _Attention
    mlp: _FeedForward | _MoE

    def __call__(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cache: _KVCache,
        *,
        cache_slot: int | None,
        cache_start: int,
        yoco_key: torch.Tensor | None = None,
        yoco_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.input_norm(x),
            positions,
            cache,
            cache_slot=cache_slot,
            cache_start=cache_start,
            yoco_key=yoco_key,
            yoco_value=yoco_value,
        ).float()
        return x + self.mlp(self.post_attention_norm(x)).float()


@dataclass
class _MTPBlock:
    """The one shared MTP block, reused for every configured prediction depth."""

    embedding_norm: _RMSNorm
    hidden_norm: _RMSNorm
    proj: _Linear
    input_norm: _RMSNorm
    attention: _Attention
    post_attention_norm: _RMSNorm
    ffn: _FeedForward

    def __call__(
        self,
        h_prev: torch.Tensor,
        token_emb: torch.Tensor,
        positions: torch.Tensor,
        cache: _KVCache,
        *,
        cache_start: int,
        yoco_key: torch.Tensor,
        yoco_value: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embedding_norm(token_emb) + self.hidden_norm(h_prev)
        hidden = self.proj(hidden)
        hidden = hidden + self.attention(
            self.input_norm(hidden),
            positions,
            cache,
            cache_slot=None,
            cache_start=cache_start,
            yoco_key=yoco_key,
            yoco_value=yoco_value,
        ).float()
        return hidden + self.ffn(self.post_attention_norm(hidden)).float()


def _load_norm(store: _TensorStore, prefix: str, eps: float, *, affine: bool = True) -> _RMSNorm:
    weight = store.take(prefix + ".weight", dtype=torch.bfloat16, required=affine)
    return _RMSNorm(weight, eps)


def _load_qk_norm(
    store: _TensorStore,
    prefix: str,
    config: ModelConfig,
    *,
    enabled: bool,
) -> _RMSNorm | _RMSClip | _Identity:
    if not enabled:
        return _Identity()
    weight = store.take(prefix + ".weight", dtype=torch.bfloat16, required=config.qk_rms_gamma)
    if config.qk_rms_clip:
        return _RMSClip(weight, config.norm_eps, config.qk_rms_limit)
    return _RMSNorm(weight, config.norm_eps)


def _load_linear(
    store: _TensorStore,
    prefix: str,
    *,
    quantized: bool = False,
    act_quant_method: str | None = None,
    aliases: Sequence[str] = (),
) -> _Linear:
    names = (prefix, *aliases)
    selected = next((name for name in names if store.has(name + ".weight")), None)
    if selected is None:
        raise KeyError(f"checkpoint is missing linear weight for {prefix!r}")
    raw_weight = store.take(selected + ".weight")
    assert raw_weight is not None
    bias = store.take(selected + ".bias", dtype=torch.bfloat16, required=False)
    scale = bias_offset = None
    if quantized:
        clip = store.take(selected + ".weight_clip_val", dtype=torch.float32, required=False)
        if clip is None:
            clip = raw_weight.float().abs().amax(dim=-1) / 0.5
        # The training model loads ordinary projection parameters into BF16
        # module storage before its fake quantizer observes them.
        weight = _ternaryseq_weight(raw_weight.to(torch.bfloat16), clip)
        if act_quant_method == "adp_8bits":
            scale = store.take(selected + ".act_scale", dtype=torch.float32, required=False)
            bias_offset = store.take(selected + ".act_bias", dtype=torch.float32, required=False)
            if scale is None:
                scale = torch.ones(weight.shape[-1], dtype=torch.float32, device=weight.device)
            if bias_offset is None:
                bias_offset = torch.zeros(weight.shape[-1], dtype=torch.float32, device=weight.device)
    else:
        weight = raw_weight.to(torch.bfloat16)
    return _Linear(weight, bias, act_quant_method if quantized else None, scale, bias_offset)


def _load_head_linear(
    store: _TensorStore,
    prefix: str,
    *,
    num_heads: int,
    head_dim: int,
    quantized: bool = False,
    act_quant_method: str | None = None,
) -> _HeadLinear:
    raw_weight = store.take(prefix + ".weight")
    assert raw_weight is not None
    if raw_weight.ndim == 2:
        raw_weight = raw_weight.reshape(num_heads, head_dim, raw_weight.shape[-1])
    if tuple(raw_weight.shape[:2]) != (num_heads, head_dim):
        raise ValueError(
            f"{prefix}.weight has shape {tuple(raw_weight.shape)}, expected "
            f"[{num_heads}, {head_dim}, in_features]"
        )
    bias = store.take(prefix + ".bias", dtype=torch.bfloat16, required=False)
    if bias is not None:
        bias = bias.reshape(-1)
    scale = bias_offset = None
    if quantized:
        clip = store.take(prefix + ".weight_clip_val", dtype=torch.float32, required=False)
        if clip is None:
            clip = raw_weight.float().abs().amax(dim=-1) / 0.5
        clip = clip.reshape(num_heads, head_dim)
        weight = _ternaryseq_weight(raw_weight.to(torch.bfloat16), clip).reshape(
            -1, raw_weight.shape[-1]
        )
        if act_quant_method == "adp_8bits":
            scale = store.take(prefix + ".act_scale", dtype=torch.float32, required=False)
            bias_offset = store.take(prefix + ".act_bias", dtype=torch.float32, required=False)
            if scale is None:
                scale = torch.ones(weight.shape[-1], dtype=torch.float32, device=weight.device)
            if bias_offset is None:
                bias_offset = torch.zeros(weight.shape[-1], dtype=torch.float32, device=weight.device)
    else:
        weight = raw_weight.to(torch.bfloat16).reshape(-1, raw_weight.shape[-1])
    return _HeadLinear(
        _Linear(weight, bias, act_quant_method if quantized else None, scale, bias_offset),
        num_heads,
        head_dim,
    )


def _load_ffn(store: _TensorStore, prefix: str, config: ModelConfig, *, quantized: bool) -> _FeedForward:
    return _FeedForward(
        _load_linear(
            store, prefix + ".up_proj", quantized=quantized, act_quant_method=config.act_quant_method
        ),
        _load_linear(
            store, prefix + ".gate_proj", quantized=quantized, act_quant_method=config.act_quant_method
        ),
        _load_linear(
            store, prefix + ".down_proj", quantized=quantized, act_quant_method=config.act_quant_method
        ),
        config.swiglu_limit,
    )


def _load_moe(store: _TensorStore, prefix: str, config: ModelConfig) -> _MoE:
    router_weight = store.take(prefix + ".gate.weight", dtype=torch.float32)
    assert router_weight is not None
    raw_w13 = store.take(prefix + ".w13")
    raw_w2 = store.take(prefix + ".w2")
    assert raw_w13 is not None and raw_w2 is not None
    if raw_w13.ndim == 4:
        raw_w13 = raw_w13.reshape(raw_w13.shape[0], -1, raw_w13.shape[-1])
    if raw_w13.shape[1] != 2 * raw_w2.shape[2] or raw_w13.shape[2] != raw_w2.shape[1]:
        raise ValueError(f"{prefix} has incompatible packed w13/w2 shapes")

    quantized = config.moe_expert_bit_linear
    w13_scale = w13_bias = w2_scale = w2_bias = None
    if quantized:
        clip_w13 = store.take(prefix + ".w13_weight_clip_val", dtype=torch.float32, required=False)
        clip_w2 = store.take(prefix + ".w2_weight_clip_val", dtype=torch.float32, required=False)
        if clip_w13 is None:
            clip_w13 = raw_w13.float().abs().amax(dim=-1) / 0.5
        if clip_w2 is None:
            clip_w2 = raw_w2.float().abs().amax(dim=-1) / 0.5
        if clip_w13.ndim == 3:
            clip_w13 = clip_w13.reshape(clip_w13.shape[0], -1)
        # Match the BF16 parameter storage used by the original inference
        # model before applying TernarySEQ in FP32.
        w13 = _ternaryseq_weight(raw_w13.to(torch.bfloat16), clip_w13)
        w2 = _ternaryseq_weight(raw_w2.to(torch.bfloat16), clip_w2)
        w13_scale = store.take(prefix + ".w13_act_scale", dtype=torch.float32, required=False)
        w13_bias = store.take(prefix + ".w13_act_bias", dtype=torch.float32, required=False)
        w2_scale = store.take(prefix + ".w2_act_scale", dtype=torch.float32, required=False)
        w2_bias = store.take(prefix + ".w2_act_bias", dtype=torch.float32, required=False)
        expert_dim, ffn_dim = w13.shape[-1], w2.shape[-1]
        device = w13.device
        if w13_scale is None:
            w13_scale = torch.ones((w13.shape[0], expert_dim), dtype=torch.float32, device=device)
        if w13_bias is None:
            w13_bias = torch.zeros((w13.shape[0], expert_dim), dtype=torch.float32, device=device)
        if w2_scale is None:
            w2_scale = torch.ones((w2.shape[0], ffn_dim), dtype=torch.float32, device=device)
        if w2_bias is None:
            w2_bias = torch.zeros((w2.shape[0], ffn_dim), dtype=torch.float32, device=device)
    else:
        w13, w2 = raw_w13.to(torch.bfloat16), raw_w2.to(torch.bfloat16)

    fc1 = fc2 = None
    fc1_norm: _RMSNorm | _Identity = _Identity()
    fc2_norm: _RMSNorm | _Identity = _Identity()
    if config.moe_latent_dim > 0:
        fc1 = _load_linear(store, prefix + ".fc1_latent_proj")
        fc2 = _load_linear(store, prefix + ".fc2_latent_proj")
        if config.moe_latent_norm:
            fc1_norm = _load_norm(store, prefix + ".fc1_latent_norm", config.norm_eps)
            fc2_norm = _load_norm(store, prefix + ".fc2_latent_norm", config.norm_eps)

    shared = shared_gate = None
    if config.d_shared_expert > 0:
        shared = _load_ffn(
            store,
            prefix + ".shared",
            config,
            quantized="ffn" in config.bit_linear_targets,
        )
        shared_gate = _load_linear(store, prefix + ".shared_gate")

    return _MoE(
        _Router.from_weight(router_weight),
        w13,
        w2,
        config.moe_top_k,
        config.swiglu_limit,
        quantized,
        w13_scale,
        w13_bias,
        w2_scale,
        w2_bias,
        fc1,
        fc2,
        fc1_norm,
        fc2_norm,
        shared,
        shared_gate,
    )


def _load_attention(
    store: _TensorStore,
    prefix: str,
    config: ModelConfig,
    *,
    is_self_layer: bool,
    is_mtp_layer: bool = False,
) -> _Attention:
    heads = config.head if is_self_layer else config.cross_head
    use_diff = config.diff_v2 or config.diff_v3
    q_heads = heads * 2 if use_diff else heads
    targets = config.bit_linear_targets
    qkv_quantized = "attn_qkv" in targets
    q_proj = _load_head_linear(
        store,
        prefix + ".q_proj",
        num_heads=q_heads,
        head_dim=config.head_dim,
        quantized=qkv_quantized,
        act_quant_method=config.act_quant_method,
    )
    k_proj = v_proj = None
    if is_self_layer:
        k_proj = _load_head_linear(
            store,
            prefix + ".k_proj",
            num_heads=config.kv_head,
            head_dim=config.head_dim,
            quantized=qkv_quantized,
            act_quant_method=config.act_quant_method,
        )
        v_proj = _load_head_linear(
            store,
            prefix + ".v_proj",
            num_heads=config.kv_head,
            head_dim=config.head_dim,
            quantized=qkv_quantized,
            act_quant_method=config.act_quant_method,
        )

    qk_enabled = config.qk_rms_clip or (config.qk_norm and is_self_layer)
    q_norm = _load_qk_norm(store, prefix + ".q_norm", config, enabled=qk_enabled)
    k_norm = _load_qk_norm(store, prefix + ".k_norm", config, enabled=qk_enabled and is_self_layer)
    o_proj = _load_linear(
        store,
        prefix + ".o_proj",
        quantized="attn_out" in targets,
        act_quant_method=config.act_quant_method,
    )
    gate_proj = None
    if use_diff or config.headwise_glu:
        gate_heads = 2 * heads if config.diff_v3 else heads
        gate_proj = _load_linear(
            store,
            prefix + ".gate_proj",
            quantized="attn_gate" in targets,
            act_quant_method=config.act_quant_method,
            aliases=(prefix + ".diff_gate_proj",),
        )
        if gate_proj.weight.shape[0] != gate_heads:
            raise ValueError(f"{prefix}.gate_proj has unexpected output width")

    inv_freq = None
    window_size = -1
    if is_mtp_layer:
        # Main's MTP Attention is a cross-decoder attention module which still
        # uses the YOCO local window over the shared global KV cache.
        window_size = config.yoco_window_size
    elif is_self_layer:
        if config.yoco_cross_layers > 0:
            window_size = config.yoco_window_size
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=store.device)
                / config.head_dim
            )
        )
    return _Attention(
        is_self_layer,
        q_proj,
        k_proj,
        v_proj,
        q_norm,
        k_norm,
        o_proj,
        gate_proj,
        config.diff_v2,
        config.diff_v3,
        config.headwise_glu,
        window_size,
        inv_freq,
    )


def _load_mtp_block(store: _TensorStore, config: ModelConfig) -> _MTPBlock:
    """Load the single MTP block whose weights are shared across all depths."""

    prefix = "mtp_block"
    return _MTPBlock(
        _load_norm(store, prefix + ".embedding_norm", config.norm_eps),
        _load_norm(store, prefix + ".hidden_norm", config.norm_eps),
        _load_linear(store, prefix + ".proj"),
        _load_norm(store, prefix + ".input_norm", config.norm_eps),
        _load_attention(
            store,
            prefix + ".self_attn",
            config,
            is_self_layer=False,
            is_mtp_layer=True,
        ),
        _load_norm(store, prefix + ".post_attn_norm", config.norm_eps),
        _load_ffn(
            store,
            prefix + ".ffn",
            config,
            quantized="ffn" in config.bit_linear_targets,
        ),
    )


class QuantMoEInferenceModel:
    """Single-request native-PyTorch YOCO model loaded from a merged checkpoint."""

    def __init__(self, config: ModelConfig, store: _TensorStore):
        self.config = config
        self.device = store.device
        self.mtp_block: _MTPBlock | None = None
        self.mtp_norm: _RMSNorm | None = None
        self.mtp_embedding_weight: torch.Tensor | None = None
        if config.quant_mode != "bfloat16":
            warnings.warn(
                f"quant_mode={config.quant_mode!r} has no external kernel dependency in this "
                "reference; non-BitLinear projections run in BF16",
                RuntimeWarning,
                stacklevel=2,
            )

        embedding = store.take("tok_embeddings.weight")
        assert embedding is not None
        embedding_bf16 = embedding.to(torch.bfloat16)
        self.embedding_weight = (
            _blockwise_int8_weight(embedding_bf16, config.int8_embedding_block_size)
            if config.int8_embedding
            else embedding_bf16
        )

        self.emb_proj_in = self.emb_proj_out = None
        self.emb_in_norm = self.emb_out_norm = None
        if config.embproj:
            self.emb_proj_in = _load_linear(store, "emb_proj_in")
            self.emb_proj_out = _load_linear(store, "emb_proj_out")
            if config.emb_proj_norm:
                self.emb_in_norm = _load_norm(store, "emb_in_norm", config.norm_eps)
                self.emb_out_norm = _load_norm(store, "emb_out_norm", config.norm_eps)

        self.layers: list[_Block] = []
        self_layer_count = config.n_layers - config.yoco_cross_layers
        for index in range(config.n_layers):
            prefix = f"layers.{index}"
            is_self_layer = index < self_layer_count
            attention = _load_attention(
                store, prefix + ".self_attn", config, is_self_layer=is_self_layer
            )
            is_moe = config.moe and index >= config.dense_layers
            mlp: _FeedForward | _MoE
            if is_moe:
                mlp = _load_moe(store, prefix + ".mlp", config)
            else:
                mlp = _load_ffn(
                    store,
                    prefix + ".mlp",
                    config,
                    quantized="ffn" in config.bit_linear_targets,
                )
            self.layers.append(
                _Block(
                    _load_norm(store, prefix + ".input_layernorm", config.norm_eps),
                    _load_norm(store, prefix + ".post_attention_layernorm", config.norm_eps),
                    attention,
                    mlp,
                )
            )

        self.norm = _load_norm(store, "norm", config.norm_eps)
        self.yoco_norm = self.yoco_k_proj = self.yoco_v_proj = self.yoco_k_norm = None
        if config.yoco_cross_layers > 0:
            self.yoco_norm = _load_norm(store, "yoco_norm", config.norm_eps)
            yoco_quantized = "yoco_kv" in config.bit_linear_targets
            self.yoco_k_proj = _load_head_linear(
                store,
                "k_proj",
                num_heads=config.cross_kv_head,
                head_dim=config.head_dim,
                quantized=yoco_quantized,
                act_quant_method=config.act_quant_method,
            )
            self.yoco_v_proj = _load_head_linear(
                store,
                "v_proj",
                num_heads=config.cross_kv_head,
                head_dim=config.head_dim,
                quantized=yoco_quantized,
                act_quant_method=config.act_quant_method,
            )
            if config.qk_rms_clip:
                self.yoco_k_norm = _load_qk_norm(store, "k_norm", config, enabled=True)
            else:
                self.yoco_k_norm = _Identity()

        raw_output = store.take("output.weight", required=False)
        if raw_output is None:
            if not config.weight_tying:
                raise KeyError("checkpoint is missing required tensor 'output.weight'")
            raw_output = embedding_bf16
        output_source = raw_output.to(torch.bfloat16)
        output_weight = (
            _blockwise_int8_weight(output_source, config.int8_lm_head_block_size)
            if config.int8_lm_head
            else output_source
        )
        self.output = _Linear(output_weight)
        if config.mtp_n:
            # Main uses output.weight as the source of MTP token embeddings,
            # then applies the embedding quantizer (which can differ from the
            # LM-head block size) before the optional embproj input mapping.
            self.mtp_embedding_weight = (
                _blockwise_int8_weight(output_source, config.int8_embedding_block_size)
                if config.int8_embedding
                else output_source
            )
            self.mtp_block = _load_mtp_block(store, config)
            self.mtp_norm = _load_norm(store, "mtp_norm", config.norm_eps)
        self.cache = _KVCache(
            self_layers=self_layer_count * config.universal_loop,
            max_seq_len=config.max_seq_len,
            kv_head=config.kv_head,
            cross_kv_head=config.cross_kv_head,
            head_dim=config.head_dim,
            has_yoco_cache=config.yoco_cross_layers > 0,
            device=self.device,
        )
        # Discard remaining optimizer and auxiliary entries. Everything needed
        # for base and optional MTP inference is now owned by this instance.
        store.clear()
        # Loading an FP checkpoint and materializing inference-only weights
        # creates short-lived CUDA allocations. Release only those free blocks;
        # live model weights and KV cache remain owned by this instance.
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def reset_cache(self) -> None:
        self.cache.reset()

    def _project_input_embedding(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.emb_proj_in is not None:
            if self.emb_in_norm is not None:
                hidden = self.emb_in_norm(hidden)
            hidden = self.emb_proj_in(hidden).float()
        return hidden

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._project_input_embedding(
            self.embedding_weight.index_select(0, input_ids).float()
        )

    def _embed_mtp(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.mtp_embedding_weight is not None
        return self._project_input_embedding(
            self.mtp_embedding_weight.index_select(0, input_ids).float()
        )

    def _project_output_embedding(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.emb_proj_out is not None:
            hidden = self.emb_proj_out(hidden)
            if self.emb_out_norm is not None:
                hidden = self.emb_out_norm(hidden)
        return hidden

    def _project_output_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._project_output_embedding(self.norm(hidden))

    def _normalize_mtp_input_ids(
        self,
        mtp_input_ids: Sequence[Sequence[int] | torch.Tensor] | torch.Tensor,
        *,
        token_count: int,
    ) -> list[torch.Tensor]:
        if self.config.mtp_n <= 0:
            raise ValueError("MTP input IDs were supplied but mtp_n is zero")
        if isinstance(mtp_input_ids, torch.Tensor):
            if mtp_input_ids.ndim == 1:
                if self.config.mtp_n != 1:
                    raise ValueError("one-dimensional MTP input IDs require mtp_n=1")
                values = [mtp_input_ids]
            elif mtp_input_ids.ndim == 2:
                if mtp_input_ids.shape[0] != self.config.mtp_n:
                    raise ValueError(
                        f"expected {self.config.mtp_n} MTP depths, got {mtp_input_ids.shape[0]}"
                    )
                values = [mtp_input_ids[index] for index in range(self.config.mtp_n)]
            else:
                raise ValueError("MTP input IDs must be a [depth, tokens] tensor")
        else:
            if len(mtp_input_ids) != self.config.mtp_n:
                raise ValueError(
                    f"expected {self.config.mtp_n} MTP depths, got {len(mtp_input_ids)}"
                )
            values = list(mtp_input_ids)

        result = [
            torch.as_tensor(value, dtype=torch.long, device=self.device).reshape(-1)
            for value in values
        ]
        for index, value in enumerate(result):
            if value.numel() != token_count:
                raise ValueError(
                    f"MTP depth {index} has {value.numel()} tokens; expected {token_count}"
                )
        return result

    def _run_mtp(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_start: int,
        mtp_input_ids: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        assert self.mtp_block is not None and self.mtp_norm is not None
        assert self.cache.yoco_key is not None and self.cache.yoco_value is not None
        mtp_hidden = hidden
        logits: list[torch.Tensor] = []
        for input_ids in mtp_input_ids:
            mtp_hidden = self.mtp_block(
                mtp_hidden,
                self._embed_mtp(input_ids),
                positions,
                self.cache,
                cache_start=cache_start,
                yoco_key=self.cache.yoco_key,
                yoco_value=self.cache.yoco_value,
            )
            logits.append(
                self.output(self._project_output_embedding(self.mtp_norm(mtp_hidden))).float()
            )
        return logits

    @torch.inference_mode()
    def _forward_tokens(
        self,
        input_ids: torch.Tensor,
        *,
        return_mtp_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if input_ids.ndim != 1 or input_ids.numel() == 0:
            raise ValueError("input_ids must be a non-empty one-dimensional token tensor")
        start = self.cache.length
        end = start + input_ids.numel()
        if end > self.config.max_seq_len:
            raise ValueError(
                f"requested sequence length {end} exceeds max_seq_len={self.config.max_seq_len}"
            )
        positions = torch.arange(start, end, dtype=torch.long, device=self.device)
        hidden = self._embed(input_ids)

        self_layer_count = self.config.n_layers - self.config.yoco_cross_layers
        cache_slot = 0
        for _ in range(self.config.universal_loop):
            for layer in self.layers[:self_layer_count]:
                hidden = layer(
                    hidden,
                    positions,
                    self.cache,
                    cache_slot=cache_slot,
                    cache_start=start,
                )
                cache_slot += 1

        if self.config.yoco_cross_layers > 0:
            assert self.yoco_norm is not None
            assert self.yoco_k_proj is not None and self.yoco_v_proj is not None
            assert self.yoco_k_norm is not None
            yoco_hidden = self.yoco_norm(hidden)
            yoco_key = self.yoco_k_norm(self.yoco_k_proj(yoco_hidden))
            yoco_value = self.yoco_v_proj(yoco_hidden)
            self.cache.append_yoco(start, yoco_key, yoco_value)
            assert self.cache.yoco_key is not None and self.cache.yoco_value is not None
            for layer in self.layers[self_layer_count:]:
                hidden = layer(
                    hidden,
                    positions,
                    self.cache,
                    cache_slot=None,
                    cache_start=start,
                    yoco_key=self.cache.yoco_key,
                    yoco_value=self.cache.yoco_value,
                )

        self.cache.length = end
        logits = self.output(self._project_output_hidden(hidden)).float()
        if return_mtp_context:
            return logits, hidden, positions, start
        return logits

    @torch.inference_mode()
    def prefill(self, input_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Reset the request cache and return logits for every prompt token."""
        tokens = torch.as_tensor(input_ids, dtype=torch.long, device=self.device).reshape(-1)
        self.reset_cache()
        return self._forward_tokens(tokens)

    @torch.inference_mode()
    def decode(self, input_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Append one or more token IDs to the existing request cache."""
        tokens = torch.as_tensor(input_ids, dtype=torch.long, device=self.device).reshape(-1)
        return self._forward_tokens(tokens)

    @torch.inference_mode()
    def prefill_with_mtp(
        self,
        input_ids: Sequence[int] | torch.Tensor,
        mtp_input_ids: Sequence[Sequence[int] | torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Reset the cache and return base plus every shared-depth MTP logit tensor."""

        tokens = torch.as_tensor(input_ids, dtype=torch.long, device=self.device).reshape(-1)
        mtp_ids = self._normalize_mtp_input_ids(mtp_input_ids, token_count=tokens.numel())
        self.reset_cache()
        logits, hidden, positions, start = self._forward_tokens(
            tokens, return_mtp_context=True
        )
        return logits, self._run_mtp(hidden, positions, start, mtp_ids)

    @torch.inference_mode()
    def decode_with_mtp(
        self,
        input_ids: Sequence[int] | torch.Tensor,
        mtp_input_ids: Sequence[Sequence[int] | torch.Tensor] | torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Append tokens and return base plus every shared-depth MTP logit tensor."""

        tokens = torch.as_tensor(input_ids, dtype=torch.long, device=self.device).reshape(-1)
        mtp_ids = self._normalize_mtp_input_ids(mtp_input_ids, token_count=tokens.numel())
        logits, hidden, positions, start = self._forward_tokens(
            tokens, return_mtp_context=True
        )
        return logits, self._run_mtp(hidden, positions, start, mtp_ids)

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Sequence[int] | torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 0.95,
        eos_token_id: int | None = None,
    ) -> list[int]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        logits = self.prefill(prompt_ids)[-1]
        generated: list[int] = []
        for index in range(max_new_tokens):
            token = _sample_next(logits, temperature, top_p)
            token_id = int(token.item())
            if eos_token_id is not None and token_id == eos_token_id:
                break
            generated.append(token_id)
            if index + 1 < max_new_tokens:
                logits = self.decode([token_id])[-1]
        return generated


def _sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_indices = probabilities.sort(descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs <= top_p
        sorted_probs = sorted_probs * keep
        probabilities = torch.zeros_like(probabilities).scatter(0, sorted_indices, sorted_probs)
        probabilities.div_(probabilities.sum())
    return torch.multinomial(probabilities, num_samples=1).squeeze(0)


def load_model(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device = "cuda",
    fallback_model: str | None = None,
    quant_moe: bool | None = None,
    int8_embedding: bool | None = None,
    int8_lm_head: bool | None = None,
    mtp_n: int | None = None,
) -> QuantMoEInferenceModel:
    """Load a single-GPU reference model with no training-stack dependencies."""
    checkpoint_dir = Path(checkpoint_dir)
    target_device = torch.device(device)
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device requests CUDA but torch.cuda.is_available() is false")
        if target_device.index is not None:
            torch.cuda.set_device(target_device)
    config, _ = load_model_args(checkpoint_dir, fallback_model)
    apply_inference_quant_overrides(
        config,
        quant_moe=quant_moe,
        int8_embedding=int8_embedding,
        int8_lm_head=int8_lm_head,
        mtp_n=mtp_n,
    )
    state = load_model_state(checkpoint_dir)
    return QuantMoEInferenceModel(config, _TensorStore(state, target_device))


def _parse_input_ids(value: str) -> list[int]:
    value = value.strip()
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("--input-ids JSON value must be a list")
        return [int(token) for token in parsed]
    return [int(token) for token in value.replace(",", " ").split()]


def _load_input_ids(path: Path) -> list[int]:
    if path.suffix == ".pt":
        try:
            values = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            values = torch.load(path, map_location="cpu")
        return [int(token) for token in torch.as_tensor(values).reshape(-1).tolist()]
    if path.suffix == ".json":
        return _parse_input_ids(path.read_text(encoding="utf-8"))
    return _parse_input_ids(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS))
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-ids", help="Comma/space separated IDs, or a JSON list.")
    input_group.add_argument("--input-ids-file", type=Path, help="Text, JSON, or .pt token IDs.")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--eos-token-id", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--quant-moe",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override metadata and run routed experts as ADP8 + TernarySEQ.",
    )
    parser.add_argument(
        "--int8-embedding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override metadata for blockwise INT8 embedding materialization.",
    )
    parser.add_argument(
        "--int8-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override metadata for blockwise INT8 LM-head materialization.",
    )
    parser.add_argument(
        "--mtp-n",
        type=int,
        default=None,
        help="Override metadata MTP depth; the checkpoint must contain mtp_block weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
    input_ids = _load_input_ids(args.input_ids_file) if args.input_ids_file else _parse_input_ids(args.input_ids)
    model = load_model(
        args.checkpoint_dir,
        device=device,
        fallback_model=args.model,
        quant_moe=args.quant_moe,
        int8_embedding=args.int8_embedding,
        int8_lm_head=args.int8_head,
        mtp_n=args.mtp_n,
    )
    generated = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=args.eos_token_id,
    )
    print(json.dumps({"input_ids": input_ids, "generated_ids": generated}))


if __name__ == "__main__":
    main()
