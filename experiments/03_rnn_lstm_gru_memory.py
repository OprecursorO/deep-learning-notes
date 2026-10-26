"""Compare RNN, LSTM, and GRU on a long-memory toy task.

The target is the first bit of a length-30 sequence. This is intentionally
small: the question is whether the final hidden state preserved early input.
"""
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(n=2400, length=30, seed=19):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n, length), generator=g).float().unsqueeze(-1)
    y = x[:, 0, 0].long()
    return x, y


class SequenceClassifier(nn.Module):
    def __init__(self, kind):
        super().__init__()
        cell = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.kind = kind
        self.rnn = cell(1, 24, batch_first=True)
        self.head = nn.Linear(24, 2)

    def forward(self, x):
        output, state = self.rnn(x)
        if self.kind == "lstm":
            state = state[0]
        return self.head(state[-1])


def run(kind, train_loader, valid_x, valid_y, epochs):
    model = SequenceClassifier(kind)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(valid_x).argmax(1) == valid_y).float().mean().item()
    return acc


def main(epochs):
    torch.manual_seed(19)
    x, y = make_data()
    split = int(0.8 * len(x))
    loader = DataLoader(TensorDataset(x[:split], y[:split]), batch_size=64, shuffle=True)
    for kind in ("rnn", "lstm", "gru"):
        print(f"{kind.upper():4s} validation_accuracy="
              f"{run(kind, loader, x[split:], y[split:], epochs):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    main(parser.parse_args().epochs)
