"""Multi-head attention, in PyTorch, on the sentence used throughout the tutorial.

Three things happen here:

1. `multi_head_attention` runs `h` heads over the same sentence, concatenates
    their outputs and projects them through W_O, exactly as section 07 defines.
2. `check_figure_numbers` re-derives the three weight matrices drawn in
    `heads_ask_differently` from the raw scores behind them, so the figure and
    the code cannot silently disagree.
3. `check_concat_equals_sum_of_heads` verifies the identity claimed in the aside,
    that concatenating and projecting is the same as letting every head write
    its own full-width vector and adding them up.

Run:  uv run python topics/multi_head.py
"""

import torch

WORDS = ["The", "cat", "chased", "the", "mouse"]

# Raw QK^T for each of the three heads in the figure, before any scaling.
# Head 1 is the matrix section 05 has been using all along.
RAW_SCORES = torch.tensor(
    [
        [  # head 1: chased puts its weight on mouse, its object
            [0.90, 2.80, -0.70, -1.60, 0.20],
            [0.40, 1.10, 2.40, -0.50, 0.85],
            [-3.20, -0.59, -1.30, -2.49, 1.49],
            [-0.80, 0.30, 0.60, 0.70, 2.60],
            [-1.20, 0.80, 2.20, -0.60, 1.30],
        ],
        [  # head 2: chased puts its weight on cat, its subject
            [0.30, 2.00, -0.40, -1.00, 0.10],
            [1.60, 0.90, 0.20, -0.60, -0.30],
            [-1.10, 2.55, -0.30, -0.90, -0.05],
            [-0.50, 1.40, 0.30, 0.40, 0.60],
            [-0.60, 2.30, 0.90, -0.20, 0.50],
        ],
        [  # head 3: every word keeps most of itself
            [2.40, -0.30, -0.80, 0.60, -1.00],
            [-0.40, 2.30, 0.30, -0.70, 0.20],
            [-0.90, 0.20, 2.50, -0.50, 0.40],
            [0.50, -0.60, -0.20, 2.40, 0.30],
            [-0.80, 0.10, 0.40, -0.30, 2.60],
        ],
    ]
)

# The weights those scores become, as printed in the figure.
EXPECTED_WEIGHTS = torch.tensor(
    [
        [
            [0.19, 0.57, 0.07, 0.04, 0.13],
            [0.13, 0.20, 0.42, 0.08, 0.17],
            [0.04, 0.18, 0.12, 0.06, 0.60],
            [0.07, 0.13, 0.15, 0.16, 0.49],
            [0.06, 0.19, 0.42, 0.08, 0.25],
        ],
        [
            [0.18, 0.47, 0.12, 0.08, 0.16],
            [0.37, 0.24, 0.16, 0.10, 0.12],
            [0.07, 0.60, 0.12, 0.08, 0.13],
            [0.11, 0.33, 0.17, 0.18, 0.21],
            [0.08, 0.45, 0.20, 0.11, 0.16],
        ],
        [
            [0.54, 0.11, 0.08, 0.19, 0.08],
            [0.11, 0.50, 0.16, 0.09, 0.15],
            [0.07, 0.14, 0.53, 0.09, 0.16],
            [0.16, 0.09, 0.11, 0.49, 0.15],
            [0.08, 0.13, 0.15, 0.10, 0.54],
        ],
    ]
)


def multi_head_attention(X, W_q, W_k, W_v, W_o):
    """
    h heads of self-attention over the same X. The head weight matrices carry a
    leading head dimension, so W_q is (h, d_model, d_k) rather than one matrix.

    Dimensions:

      h        --> number of heads, the lookups run in parallel
      n        --> sequence length, one row per word, 5 for our sentence
      d_model  --> width a word arrives and leaves at
      d_k      --> width of a query and of a key inside one head, d_model // h
      d_v      --> width of a value inside one head, also d_model // h

    Shapes in:

      X        (n, d_model)
      W_q      (h, d_model, d_k)      one projection per head
      W_k      (h, d_model, d_k)
      W_v      (h, d_model, d_v)
      W_o      (d_model, d_model)     the only place the heads meet

    Returns the output (n, d_model) and the weights (h, n, n).
    """
    # Every head projects the same X, and no head sees another head's work
    Q = X @ W_q  # (h, n, d_k)
    K = X @ W_k  # (h, n, d_k)
    V = X @ W_v  # (h, n, d_v)

    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1)  # (h, n, n)
    weights = torch.softmax(scores / d_k**0.5, dim=-1)
    heads = weights @ V  # (h, n, d_v)

    # write the h answers side by side, then let W_o mix them
    concat = torch.cat([heads[m] for m in range(heads.shape[0])], dim=-1)

    return concat @ W_o, weights


def demo():
    torch.manual_seed(0)
    n, d_model, h = len(WORDS), 9, 3
    d_k = d_v = d_model // h  # splitting the width

    X = torch.randn(n, d_model)
    W_q = torch.randn(h, d_model, d_k)
    W_k = torch.randn(h, d_model, d_k)
    W_v = torch.randn(h, d_model, d_v)
    W_o = torch.randn(d_model, d_model)

    out, weights = multi_head_attention(X, W_q, W_k, W_v, W_o)

    print("shapes")
    print(f"  X        {tuple(X.shape)}      n x d_model")
    print(f"  W_q      {tuple(W_q.shape)}   h x d_model x d_k")
    print(f"  weights  {tuple(weights.shape)}   h x n x n")
    print(f"  output   {tuple(out.shape)}      n x d_model")

    # each head holds its own budget, and every one of them sums to 1
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(h, n), atol=1e-5), row_sums
    print(f"\nall {h * n} rows of weights sum to 1, one budget per head per word")

    # a word leaves at the width it arrived at
    assert out.shape == X.shape
    print("the output is as wide as the input, because h * d_v = d_model")

    # h heads at d_model/h cost what one head at full width cost
    per_head = W_q.numel() + W_k.numel() + W_v.numel()
    full_width = 3 * d_model * d_model
    assert per_head == full_width, (per_head, full_width)
    print(
        f"\nthe {h} heads hold {per_head} numbers, a single full-width head "
        f"holds {full_width}"
    )
    print(f"W_o adds {W_o.numel()} more, so re-joining the heads is what costs")

    # the one line of the bill that does grow with h
    print(f"but the weights are {h} tables of {n} x {n}, where one head kept 1")


def check_figure_numbers():
    """The figure claims these scores give these weights. Do they?"""
    d_k = 3
    weights = torch.softmax(RAW_SCORES / d_k**0.5, dim=-1)

    assert torch.allclose(weights, EXPECTED_WEIGHTS, atol=5e-3), weights
    print("\nrow (3), chased, in each head:")
    for m, w in enumerate(weights, start=1):
        row = "  ".join(f"{x:.2f}" for x in w[2])
        print(f"  head {m}   {row}")
    print("0.60 on mouse, 0.60 on cat, 0.53 on chased itself, each out of its own 1")


def check_concat_equals_sum_of_heads():
    """The aside claims Concat(heads) W_o == sum_m head_m W_o^(m). Does it?"""
    torch.manual_seed(1)
    n, d_model, h = len(WORDS), 9, 3
    d_v = d_model // h

    heads = torch.randn(h, n, d_v)
    W_o = torch.randn(d_model, d_model)

    concat = torch.cat([heads[m] for m in range(h)], dim=-1)  # (n, h * d_v)
    joined = concat @ W_o

    # cut W_o into h blocks of d_v rows, one block per head
    blocks = W_o.view(h, d_v, d_model)
    summed = sum(heads[m] @ blocks[m] for m in range(h))

    assert torch.allclose(joined, summed, atol=1e-5), (joined - summed).abs().max()
    print("\nconcatenate-then-project equals the sum of h full-width writes")
    print(
        f"each of the {h} terms is {tuple((heads[0] @ blocks[0]).shape)}, "
        "the full width, so no head owns a slice of the answer"
    )


if __name__ == "__main__":
    demo()
    check_figure_numbers()
    check_concat_equals_sum_of_heads()
