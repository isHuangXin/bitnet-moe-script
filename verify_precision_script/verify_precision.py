"""Precision verification: compare TernarySEQ (per-row) vs I2_S (per-tensor) quantization.

This script verifies the numerical difference between:
- Path A: TernarySEQ quantization (as in simple_quant_moe_infer_with_mtp.py)
- Path B: I2_S quantization (as in BitNet/llama.cpp engine)

Both operate on the same random weight matrices with dimensions matching
the YOCO-MoE-30B-A3B-V3 model.
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
    """TernarySEQ: per-row quantization (reference implementation from simple_quant_moe_infer)."""
    alpha = clip.float().clamp_min(TERNARY_EPS).unsqueeze(-1)
    q = weight.float()
    q = q / alpha
    q = q.clamp(-TERNARY_CLIP_RATIO, TERNARY_CLIP_RATIO)
    q = (q * TERNARY_LEVELS).round() / TERNARY_LEVELS
    q = q * alpha
    return q


def quantize_i2s(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    """I2_S: per-tensor quantization (as in BitNet/llama.cpp).

    Returns: (dequantized_weight, scale)
    """
    w = weight.float().flatten()
    # Scale = first nonzero abs value (matches C quantize_i2_s)
    nonzero = w.abs()
    nonzero = nonzero[nonzero > 1e-6]
    if len(nonzero) > 0:
        scale = nonzero[0].item()
    else:
        scale = 1e-5

    # Quantize to ternary
    inv_scale = 1.0 / scale
    ternary = (w * inv_scale).round().clamp(-1, 1)

    # Dequantize
    dequant = ternary * scale
    return dequant.reshape(weight.shape), scale


def generate_bitlinear_weight(M: int, K: int, seed: int = 42) -> torch.Tensor:
    """Generate a random weight that has been 'trained' with BitLinear.

    In a real trained model, weights would already be near-ternary.
    We simulate this by generating and then applying TernarySEQ.
    """
    torch.manual_seed(seed)
    # Random BF16 weight (as if from training)
    w = torch.randn(M, K, dtype=torch.bfloat16)
    return w


def compute_clip_per_row(weight: torch.Tensor) -> torch.Tensor:
    """Compute per-row clip values (max abs per row) for TernarySEQ."""
    return weight.float().abs().amax(dim=-1)


def compare_matmul(name: str, M: int, K: int, seed: int = 42):
    """Compare matmul output using TernarySEQ vs I2_S quantized weights."""
    # Generate weight
    W = generate_bitlinear_weight(M, K, seed)

    # Generate activation (batch=1)
    torch.manual_seed(seed + 1000)
    x = torch.randn(1, K, dtype=torch.bfloat16)

    # Path A: TernarySEQ (per-row)
    clip = compute_clip_per_row(W)
    W_ternaryseq = quantize_ternaryseq(W, clip)
    out_A = F.linear(x.float(), W_ternaryseq.float()).squeeze(0)

    # Path B: I2_S (per-tensor)
    W_i2s, scale = quantize_i2s(W)
    out_B = F.linear(x.float(), W_i2s.float()).squeeze(0)

    # Compare
    diff = (out_A - out_B).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(out_A.unsqueeze(0), out_B.unsqueeze(0)).item()

    # Relative error
    rel_err = (diff / (out_A.abs() + 1e-8)).mean().item()

    # Also check how many ternary values differ
    clip_expanded = clip.unsqueeze(-1)
    alpha = clip_expanded.float().clamp_min(TERNARY_EPS)
    t_seq = (W.float() / alpha).clamp(-TERNARY_CLIP_RATIO, TERNARY_CLIP_RATIO)
    t_seq = (t_seq * TERNARY_LEVELS).round() / TERNARY_LEVELS  # {-1, 0, 1} (in 1/1.5 units)

    w_flat = W.float().flatten()
    nonzero = w_flat.abs()
    nz_vals = nonzero[nonzero > 1e-6]
    i2s_scale = nz_vals[0].item() if len(nz_vals) > 0 else 1e-5
    t_i2s = (w_flat / i2s_scale).round().clamp(-1, 1).reshape(M, K)

    # Compare ternary assignments
    # TernarySEQ produces values in {-1/1.5, 0, 1/1.5} = {-0.667, 0, 0.667}
    # I2_S produces {-1, 0, 1}
    # Normalize both to {-1, 0, 1} for comparison
    t_seq_norm = (t_seq * TERNARY_LEVELS).round()  # back to {-1, 0, 1}
    ternary_match = (t_seq_norm == t_i2s).float().mean().item()

    return {
        "name": name,
        "shape": (M, K),
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "cos_sim": cos_sim,
        "rel_err": rel_err,
        "ternary_match": ternary_match,
        "i2s_scale": scale,
    }


def main():
    print("=" * 70)
    print("  Precision Verification: TernarySEQ (per-row) vs I2_S (per-tensor)")
    print("  Model: YOCO-MoE-30B-A3B-V3")
    print("=" * 70)
    print()
    print("TernarySEQ: alpha=max_abs_per_row, q=round(W/alpha*1.5)/1.5*alpha")
    print("I2_S:       scale=first_nonzero_abs, q=round(W/scale)*scale")
    print()

    results = []
    for name, (M, K) in DIMS.items():
        r = compare_matmul(name, M, K)
        results.append(r)

    # Print results
    print(f"{'Layer':<12} {'Shape':<14} {'MaxDiff':<12} {'MeanDiff':<12} {'CosSim':<10} {'RelErr':<10} {'TernMatch':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<12} {str(r['shape']):<14} {r['max_diff']:<12.4f} {r['mean_diff']:<12.6f} {r['cos_sim']:<10.6f} {r['rel_err']:<10.6f} {r['ternary_match']*100:<9.1f}%")

    print()
    print("=" * 70)
    print("  Interpretation:")
    print("  - CosSim ≈ 1.0: output directions match (good)")
    print("  - TernMatch: % of weights quantized to same ternary value")
    print("  - Difference comes from per-tensor vs per-row scale granularity")
    print("=" * 70)

    # Summary
    avg_cos = np.mean([r['cos_sim'] for r in results])
    avg_match = np.mean([r['ternary_match'] for r in results])
    print(f"\n  Average cosine similarity: {avg_cos:.6f}")
    print(f"  Average ternary match:     {avg_match*100:.1f}%")

    if avg_cos > 0.95:
        print("\n  ✅ PASS: Outputs are numerically consistent (expected quantization noise)")
    elif avg_cos > 0.8:
        print("\n  ⚠️  WARN: Moderate divergence — per-tensor vs per-row scale causes notable diff")
    else:
        print("\n  ❌ FAIL: Significant divergence — possible bug")


if __name__ == "__main__":
    main()
