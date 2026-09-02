"""Self-attention, in PyTorch, on the sentence used throughout the tutorial.

Two things happen here:

  1. `attention` implements the four forward steps drawn in the section 06
     breakdown figure: project, score, weight, blend.
  2. `check_tutorial_numbers` re-derives the weights quoted in the figures
     from the raw scores quoted in the figures, so the article and the code
     cannot silently disagree.

Run:  uv run python topics/self_attention.py
"""

import torch

WORDS = ["The", "cat", "chased", "the", "mouse"]

# The raw QK^T from the section 05 figure, before any scaling. Row i holds
# word i's scores against all five keys.
RAW_SCORES = torch.tensor(
    [
        [0.90, 2.80, -0.70, -1.60, 0.20],
        [0.40, 1.10, 2.40, -0.50, 0.85],
        [-3.20, -0.59, -1.30, -2.49, 1.49],
        [-0.80, 0.30, 0.60, 0.70, 2.60],
        [-1.20, 0.80, 2.20, -0.60, 1.30],
    ]
)

# The weights those scores become, as printed in every figure since page 01.
EXPECTED_WEIGHTS = torch.tensor(
    [
        [0.19, 0.57, 0.07, 0.04, 0.13],
        [0.13, 0.20, 0.42, 0.08, 0.17],
        [0.04, 0.18, 0.12, 0.06, 0.60],
        [0.07, 0.13, 0.15, 0.16, 0.49],
        [0.06, 0.19, 0.42, 0.08, 0.25],
    ]
)


def attention(X, W_q, W_k, W_v):
    """
    One head of self-attention. Returns the output and the weights.
    """
    # (a) project the same X three ways
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # (b) every query against every key
    scores = Q @ K.transpose(-2, -1)

    # (c) scale then softmax along each row on its own
    d_k = Q.shape[-1]
    weights = torch.softmax(scores / d_k**0.5, dim=-1)

    # (d) blend the values
    return weights @ V, weights


def demo():
    torch.manual_seed(0)
    n, d_model, d_k, d_v = len(WORDS), 8, 3, 3

    X = torch.randn(n, d_model)
    W_q = torch.randn(d_model, d_k)
    W_k = torch.randn(d_model, d_k)
    W_v = torch.randn(d_model, d_v)

    out, weights = attention(X, W_q, W_k, W_v)

    print("shapes")
    print(f"  X       {tuple(X.shape)}      n x d_model")
    print(f"  W_q     {tuple(W_q.shape)}      d_model x d_k")
    print(f"  weights {tuple(weights.shape)}      n x n")
    print(f"  output  {tuple(out.shape)}      n x d_v")

    # every row of the weights is a distribution over the five words
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(n), atol=1e-5), row_sums
    print("\nevery row of the weights sums to 1, as it should")


def check_tutorial_numbers():
    """The figures claim these scores give these weights. Do they?"""
    d_k = 3
    weights = torch.softmax(RAW_SCORES / d_k**0.5, dim=-1)

    assert torch.allclose(weights, EXPECTED_WEIGHTS, atol=5e-3), weights
    print("\nweights derived from the figure's raw scores:")
    for word, row in zip(WORDS, weights):
        print("  " + word.ljust(7) + "  ".join(f"{w:.2f}" for w in row))
    print("these match the numbers printed in the figures")


if __name__ == "__main__":
    demo()
    check_tutorial_numbers()
    check_tutorial_numbers()
