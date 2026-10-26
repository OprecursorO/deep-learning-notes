"""Minimal experiment for Basic Neural Networks.

Run:
    python 01_mlp_dataset.py

The data are two noisy interleaving half-circles. The experiment makes the
training/validation split, loss, backpropagation, and generalization visible.
"""
import argparse
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_moons(n=1200, noise=0.12, seed=7):
    g = torch.Generator().manual_seed(seed)
    n0 = n // 2
    theta = torch.rand(n0, generator=g) * math.pi
    x0 = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
    x1 = torch.stack((1 - torch.cos(theta), 0.5 - torch.sin(theta)), dim=1)
    x = torch.cat((x0, x1), dim=0)
    y = torch.cat((torch.zeros(n0), torch.ones(n0))).long()
    x += noise * torch.randn(x.shape, generator=g)
    order = torch.randperm(n, generator=g)
    return x[order], y[order]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.net(x)


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += len(y)
    return correct / total


def main(epochs):
    torch.manual_seed(7)
    x, y = make_moons()
    split = int(0.8 * len(x))
    train = TensorDataset(x[:split], y[:split])
    valid = TensorDataset(x[split:], y[split:])
    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=256)
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
        if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} train_loss={loss_sum/len(train):.4f} "
                  f"train_acc={accuracy(model, train_loader):.3f} "
                  f"val_acc={accuracy(model, valid_loader):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    main(parser.parse_args().epochs)
