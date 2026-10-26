"""Minimal CNN experiment for convolution, pooling, and tensor shapes.

Run:
    python 02_cnn_shapes.py

The synthetic images contain vertical or horizontal bars. No external image
dataset is required, so the focus stays on the current CNN slides.
"""
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_images(n=1200, size=16, seed=11):
    g = torch.Generator().manual_seed(seed)
    x = 0.08 * torch.randn(n, 1, size, size, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    for i in range(n):
        pos = int(torch.randint(3, size - 3, (1,), generator=g))
        if y[i] == 0:
            x[i, 0, :, pos - 1:pos + 2] += 1.0
        else:
            x[i, 0, pos - 1:pos + 2, :] += 1.0
    return x, y


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(16 * 4 * 4, 2)

    def forward(self, x, show_shapes=False):
        for layer in self.features:
            x = layer(x)
            if show_shapes and isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                print(f"{layer.__class__.__name__:>10}: {tuple(x.shape)}")
        return self.classifier(x.flatten(1))


def main(epochs):
    torch.manual_seed(11)
    x, y = make_images()
    split = int(0.8 * len(x))
    train_loader = DataLoader(TensorDataset(x[:split], y[:split]), batch_size=64, shuffle=True)
    valid_loader = DataLoader(TensorDataset(x[split:], y[split:]), batch_size=256)
    model = TinyCNN()
    print("input:", tuple(x[:4].shape))
    model(x[:4], show_shapes=True)
    print("trainable parameters:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        if epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(xb).argmax(1) for xb, _ in valid_loader])
            print(f"epoch={epoch:03d} val_acc={(pred == y[split:]).float().mean().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    main(parser.parse_args().epochs)
