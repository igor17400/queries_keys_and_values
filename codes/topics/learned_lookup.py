"""One attention head learns to be a dictionary.

Each example is a tiny store: a few (tag, value) pairs, plus one tag to look
up. Nothing tells the model how to do this. It has a query, some keys, some
values and a loss, and it works out the rest.

Run:  uv run python topics/learned_lookup.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)

N_TAGS, N_VALUES, N_ENTRIES = 6, 6, 4
D_MODEL, D_K = 32, 16


def batch(size):
    """
    A little store, and one tag to look up in it
    """
    tags = torch.stack([torch.randperm(N_TAGS)[:N_ENTRIES] for _ in range(size)])
    values = torch.randint(0, N_VALUES, (size, N_ENTRIES))
    picked = torch.randint(0, N_ENTRIES, (size,))
    rows = torch.arange(size)

    return tags, values, tags[rows, picked], values[rows, picked], picked


class OneHead(nn.Module):
    def __init__(self):
        super().__init__()
        # an entry is one token carrying both its tag and its value
        self.tag_emb = nn.Embedding(N_TAGS, D_MODEL)
        self.value_emb = nn.Embedding(N_VALUES, D_MODEL)
        self.W_q = nn.Linear(D_MODEL, D_K, bias=False)
        self.W_k = nn.Linear(D_MODEL, D_K, bias=False)
        self.W_v = nn.Linear(D_MODEL, D_K, bias=False)
        self.readout = nn.Linear(D_K, N_VALUES)

    def forward(self, tags, values, query_tag):
        entries = self.tag_emb(tags) + self.value_emb(values)
        query = self.tag_emb(query_tag).unsqueeze(1)
        X = torch.cat([entries, query], dim=1)

        Q, K, V = self.W_q(X), self.W_k(X), self.W_v(X)
        weights = torch.softmax(Q @ K.transpose(-2, -1) / D_K**0.5, dim=-1)
        answer = (weights @ V)[:, -1]  # the query's row of the output

        return self.readout(answer), weights


def train():
    model = OneHead()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    probe = batch(1)  # one fixed example to watch
    snapshots = []

    for step in range(401):
        tags, values, query_tag, target, _ = batch(256)
        logits, _ = model(tags, values, query_tag)
        loss = nn.functional.cross_entropy(logits, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step in (0, 10, 25, 50, 100, 400):
            with torch.no_grad():
                logits, _ = model(tags, values, query_tag)
                acc = (logits.argmax(-1) == target).float().mean().item()
                _, w = model(probe[0], probe[1], probe[2])
            snapshots.append((step, loss.item(), acc, w[0, -1].tolist()))

    return model, probe, snapshots


if __name__ == "__main__":
    model, probe, snapshots = train()
    tags, values, query_tag, target, picked = probe

    print(f"store:  tags {tags[0].tolist()}  values {values[0].tolist()}")
    print(
        f"query:  tag {query_tag.item()}  ->  entry {picked.item()}, "
        f"value {target.item()}\n"
    )

    header = "   ".join(f"e{i}" for i in range(N_ENTRIES)) + "  self"
    print(f"{'step':>5}  {'loss':>6}  {'acc':>5}   {header}")
    for step, loss, acc, row in snapshots:
        cells = "  ".join(f"{x:.2f}"[1:] for x in row)
        print(f"{step:5d}  {loss:6.3f}  {acc:5.2f}   {cells}")

    correct = snapshots[-1][3][picked.item()]
    assert correct > 0.8, f"head did not learn the lookup: {snapshots[-1][3]}"
    print(
        f"\nthe trained head puts {correct:.2f} of its attention on entry "
        f"{picked.item()}, which is the one holding the answer"
    )
