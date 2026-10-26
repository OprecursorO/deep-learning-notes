"""Visualize Q/K/V attention as a learned information-reading operation.

Run:
    python 05_attention_reading.py

The query is intentionally close to one key. The heatmap should show that
the output reads mostly from the corresponding value, not from every input
equally. This isolates scaled dot-product attention without model training.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main(output: str) -> None:
    torch.manual_seed(7)
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    values = torch.tensor([[4.0, 0.0], [0.0, 2.0], [-3.0, 0.0], [0.0, -1.0]])
    queries = torch.tensor([[0.9, 0.1], [0.0, 1.0]])
    scores = queries @ keys.T / (keys.shape[-1] ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    contexts = weights @ values

    print("scores shape:", tuple(scores.shape), "weights shape:", tuple(weights.shape))
    print("attention weights:\n", weights.round(decimals=3))
    print("context vectors:\n", contexts.round(decimals=3))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    im = axes[0].imshow(weights.numpy(), cmap="magma", vmin=0, vmax=1)
    axes[0].set(title="Attention weights A", xlabel="input key", ylabel="query")
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_yticks(range(len(queries)))
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            axes[0].text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].bar(["x", "y"], contexts[0].numpy(), label="query 0")
    axes[1].bar(["x", "y"], contexts[1].numpy(), alpha=0.65, label="query 1")
    axes[1].set(title="Weighted value readout Y = AV", ylabel="value")
    axes[1].legend()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/05_attention_reading.png")
    main(parser.parse_args().output)
