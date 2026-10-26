"""Toy seq2seq experiment: reverse a sequence and inspect attention.

Run:
    python 04_seq2seq_attention.py

This demonstrates teacher forcing during training, greedy decoding during
inference, and a small additive attention module. It is not a translation
benchmark; it isolates the mechanism from vocabulary and data engineering.
"""
import argparse
import random
import torch
from torch import nn

PAD, BOS, EOS = 0, 1, 2
VOCAB = 13  # PAD/BOS/EOS plus digits 0..9


def make_batch(batch, length, device):
    tokens = torch.randint(3, VOCAB, (batch, length), device=device)
    target = torch.cat((tokens.flip(1), torch.full((batch, 1), EOS, device=device)), dim=1)
    source = torch.cat((tokens, torch.full((batch, 1), EOS, device=device)), dim=1)
    return source, target


class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.energy = nn.Linear(hidden * 2, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, query, values):
        # query: [B, 1, H], values: [B, S, H]
        q = query.expand(-1, values.size(1), -1)
        energy = torch.tanh(self.energy(torch.cat((q, values), dim=2)))
        scores = self.score(energy).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), values)
        return context, weights


class Seq2Seq(nn.Module):
    def __init__(self, emb=24, hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, emb)
        self.encoder = nn.GRU(emb, hidden, batch_first=True)
        self.attn = Attention(hidden)
        self.decoder = nn.GRU(emb + hidden, hidden, batch_first=True)
        self.output = nn.Linear(emb + hidden + hidden, VOCAB)

    def encode(self, source):
        values, hidden = self.encoder(self.embedding(source))
        return values, hidden

    def decode_step(self, token, hidden, values):
        embedded = self.embedding(token).unsqueeze(1)
        context, weights = self.attn(hidden.transpose(0, 1), values)
        out, hidden = self.decoder(torch.cat((embedded, context), dim=2), hidden)
        logits = self.output(torch.cat((embedded, context, out), dim=2)).squeeze(1)
        return logits, hidden, weights

    def forward(self, source, target, teacher_forcing=0.7):
        values, hidden = self.encode(source)
        token = torch.full((source.size(0),), BOS, dtype=torch.long, device=source.device)
        logits = []
        for t in range(target.size(1)):
            step, hidden, _ = self.decode_step(token, hidden, values)
            logits.append(step)
            token = target[:, t] if random.random() < teacher_forcing else step.argmax(1)
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def greedy(self, source, steps):
        values, hidden = self.encode(source)
        token = torch.full((source.size(0),), BOS, dtype=torch.long, device=source.device)
        outputs, maps = [], []
        for _ in range(steps):
            step, hidden, weights = self.decode_step(token, hidden, values)
            token = step.argmax(1)
            outputs.append(token)
            maps.append(weights)
        return torch.stack(outputs, 1), torch.stack(maps, 1)


def main(epochs):
    torch.manual_seed(23)
    random.seed(23)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Seq2Seq().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        source, target = make_batch(128, 5, device)
        opt.zero_grad()
        loss = loss_fn(model(source, target).reshape(-1, VOCAB), target.reshape(-1))
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % max(1, epochs // 5) == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")
    source, target = make_batch(1, 5, device)
    pred, weights = model.greedy(source, target.size(1))
    print("source tokens:", source[0].tolist())
    print("target tokens:", target[0].tolist())
    print("greedy tokens:", pred[0].tolist())
    print("attention shape [output_step, input_step]:", tuple(weights[0].shape))
    print("first output attention:", [round(v, 3) for v in weights[0, 0].tolist()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    main(parser.parse_args().epochs)
