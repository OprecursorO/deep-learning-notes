"""Estimate attention score and KV-cache pressure as sequence length grows.

Run:
    python 08_attention_system_cost.py

This is a hardware sizing experiment, not a performance benchmark. It makes
the O(N^2) score-matrix growth and linear KV-cache growth visible in the same
figure. Change dtype, layers, heads, or batch to study a design point.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def main(output: str, layers: int, batch: int, heads: int, head_dim: int, bytes_per_element: int) -> None:
    lengths = [256, 512, 1024, 2048, 4096, 8192]
    score_mib = [batch * heads * n * n * bytes_per_element / 2**20 for n in lengths]
    kv_mib = [layers * 2 * batch * heads * n * head_dim * bytes_per_element / 2**20 for n in lengths]
    for n, score, kv in zip(lengths, score_mib, kv_mib):
        print(f"N={n:5d}  score weights={score:9.1f} MiB  KV cache={kv:9.1f} MiB")

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.loglog(lengths, score_mib, "o-", label="one-layer attention weights")
    ax.loglog(lengths, kv_mib, "s-", label="all-layer KV cache")
    ax.set(xlabel="sequence length N", ylabel="memory (MiB)", title="Attention system pressure")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/08_attention_system_cost.png")
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--bytes-per-element", type=int, default=2, help="2 for fp16/bf16, 4 for fp32")
    args = parser.parse_args()
    main(args.output, args.layers, args.batch, args.heads, args.head_dim, args.bytes_per_element)
