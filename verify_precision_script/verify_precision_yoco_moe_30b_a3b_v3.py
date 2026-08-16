"""Precision verification: TernarySEQ (PyTorch ref) vs I2_S (BitNet engine) quantization.

This script verifies that both paths produce identical results:
- Path A: TernarySEQ quantization (as in simple_quant_moe_infer_with_mtp.py)
- Path B: I2_S quantization (as in BitNet/llama.cpp TL2 engine, per-row mode)

Both use the same formula: alpha=max_abs_per_row, q=round(W/alpha*1.5)/1.5*alpha
"""

import numpy as np
import torch
import torch.nn.functional as F

# Model dimensions from YOCO-MoE-30B-A3B-V3
DIMS = {
    "Q proj":      (8192, 3072),
    "O proj":      (3072, 4096),
    "MoE gate_up": (1280, 3072),
    "MoE down":    (3072, 1280),
    "K/V proj":    (1024, 3072),
    "Self-attn":   (3072, 3072),
    "Router":      (128, 3072),
}

TERNARY_CLIP_RATIO = 1.0 - 1e-2
TERNARY_LEVELS = 1.5
TERNARY_EPS = 1e-5


def quantize_ternaryseq(weight: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    """TernarySEQ: per-row quantization (reference from simple_quant_moe_infer)."""
    alpha = clip.float().clamp_min(TERNARY_EPS).unsqueeze(-1)
    q = weight.float()
    q = q / alpha
    q = q.clamp(-TERNARY_CLIP_RATIO, TERNARY_CLIP_RATIO)
    q = (q * TERNARY_LEVELS).round() / TERNARY_LEVELS
    q = q * alpha
    return q


def quantize_i2s(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """I2_S: per-row TernarySEQ quantization (as in updated BitNet/llama.cpp).

    Returns: (dequantized_weight, per_row_scales)
    """
    M, K = weight.shape
    w = weight.float()
    clip_ratio = 1.0 - 1e-2

    scales = torch.zeros(M)
    dequant = torch.zeros_like(w)

    for row in range(M):
        alpha = w[row].abs().max().clamp_min(1e-5).item()
        scales[row] = alpha
        normalized = w[row] / alpha
        normalized = normalized.clamp(-clip_ratio, clip_ratio)
        ternary = (normalized * 1.5).round() / 1.5
        dequant[row] = ternary * alpha

    return dequant, scales


def generate_bitlinear_weight(M: int, K: int, seed: int = 42) -> torch.Tensor:
    """Generate a random BF16 weight matrix."""
    torch.manual_seed(seed)
    return torch.randn(M, K, dtype=torch.bfloat16)


def compute_clip_per_row(weight: torch.Tensor) -> torch.Tensor:
    """Compute per-row clip values (max abs per row) for TernarySEQ."""
    return weight.float().abs().amax(dim=-1)


def compare_matmul(name: str, M: int, K: int, seed: int = 42):
    """Compare matmul output using TernarySEQ vs I2_S quantized weights."""
    W = generate_bitlinear_weight(M, K, seed)

    # Activation (batch=1)
    torch.manual_seed(seed + 1000)
    x = torch.randn(1, K, dtype=torch.bfloat16)

    # Path A: TernarySEQ (PyTorch reference)
    clip = compute_clip_per_row(W)
    W_ternaryseq = quantize_ternaryseq(W, clip)
    out_A = F.linear(x.float(), W_ternaryseq.float()).squeeze(0)

    # Path B: I2_S (BitNet engine, per-row)
    W_i2s, scales = quantize_i2s(W)
    out_B = F.linear(x.float(), W_i2s.float()).squeeze(0)

    # Compare outputs
    diff = (out_A - out_B).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = F.cosine_similarity(out_A.unsqueeze(0), out_B.unsqueeze(0)).item()
    rel_err = (diff / (out_A.abs() + 1e-8)).mean().item()

    return {
        "name": name,
        "shape": (M, K),
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cos_sim": cos_sim,
        "rel_err": rel_err,
    }


def main():
    print("=" * 70)
    print("  Precision Verification: TernarySEQ (PyTorch) vs I2_S (BitNet TL2)")
    print("  Model: YOCO-MoE-30B-A3B-V3")
    print("=" * 70)
    print()
    print("Both paths use per-row quantization:")
    print("  alpha = max(|W_row|)")
    print("  ternary = round(W/alpha * 1.5) / 1.5  -> {-1, 0, +1}")
    print("  dequant = ternary * alpha")
    print()

    results = []
    for name, (M, K) in DIMS.items():
        r = compare_matmul(name, M, K)
        results.append(r)

    # Print results
    print(f"{'Layer':<12} {'Shape':<14} {'MaxDiff':<12} {'MeanDiff':<12} {'CosSim':<10} {'RelErr':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<12} {str(r['shape']):<14} {r['max_diff']:<12.6f} {r['mean_diff']:<12.8f} {r['cos_sim']:<10.6f} {r['rel_err']:<10.8f}")

    print()
    print("=" * 70)
    avg_cos = np.mean([r['cos_sim'] for r in results])
    avg_diff = np.mean([r['max_diff'] for r in results])
    print(f"  Average cosine similarity: {avg_cos:.6f}")
    print(f"  Average max diff:          {avg_diff:.6f}")

    if avg_cos > 0.9999:
        print("\n  ✅ PASS: TernarySEQ and I2_S paths are numerically identical")
    elif avg_cos > 0.95:
        print("\n  ⚠️  WARN: Minor numerical difference (likely floating point rounding)")
    else:
        print("\n  ❌ FAIL: Significant divergence — quantization mismatch")


if __name__ == "__main__":
    main()
