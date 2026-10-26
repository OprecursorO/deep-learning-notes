"""Visualize positional encoding and a causal attention mask.

Run:
    python 06_position_and_causal_mask.py

Left: sinusoidal position vectors give otherwise identical token slots
different signatures. Right: a decoder query can only read current/past
positions after the upper-triangular future positions are masked.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def sinusoidal_positions(length: int, dimension: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dimension))
    encoding = torch.zeros(length, dimension)
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div)
    return encoding


def main(output: str) -> None:
    length, dimension = 12, 16
    encoding = sinusoidal_positions(length, dimension)
    raw_scores = torch.arange(length * length, dtype=torch.float32).reshape(length, length) / 30
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    masked_scores = raw_scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)
    print("position encoding shape:", tuple(encoding.shape))
    print("future positions masked:", int(mask.sum()), "of", length * length)
    print("row sums after masked softmax:", weights.sum(-1).round(decimals=4).tolist())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(encoding.T, aspect="auto", cmap="coolwarm")
    axes[0].set(title="Sinusoidal positional encoding", xlabel="position", ylabel="feature channel")
    axes[1].imshow(weights, cmap="viridis", vmin=0, vmax=float(weights.max()))
    axes[1].set(title="Causal attention weights", xlabel="key position", ylabel="query position")
    axes[1].plot([-.5, length - .5], [-.5, length - .5], color="white", linewidth=1)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/06_position_and_causal_mask.png")
    main(parser.parse_args().output)
