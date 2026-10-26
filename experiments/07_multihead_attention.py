"""Inspect per-head attention maps from PyTorch MultiheadAttention.

Run:
    python 07_multihead_attention.py

The heads start from the same token representation but use different learned
projections. The random model is not a quality benchmark; the point is to
inspect tensor shapes and see that each head owns a separate attention map.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


def main(output: str) -> None:
    torch.manual_seed(11)
    batch, tokens, model_dim, heads = 1, 8, 16, 4
    x = torch.randn(batch, tokens, model_dim)
    layer = nn.MultiheadAttention(model_dim, heads, batch_first=True)
    y, weights = layer(x, x, x, need_weights=True, average_attn_weights=False)
    print("input shape:", tuple(x.shape), "output shape:", tuple(y.shape))
    print("per-head weights shape [B,H,Q,K]:", tuple(weights.shape))
    print("each row sum:", weights[0, 0, 0].sum().item())

    fig, axes = plt.subplots(1, heads, figsize=(12, 3), constrained_layout=True)
    for h, ax in enumerate(axes):
        im = ax.imshow(weights[0, h].detach().numpy(), cmap="magma", vmin=0, vmax=float(weights.detach().max()))
        ax.set(title=f"head {h}", xlabel="key", ylabel="query")
    fig.colorbar(im, ax=axes, fraction=0.02)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/07_multihead_attention.png")
    main(parser.parse_args().output)
