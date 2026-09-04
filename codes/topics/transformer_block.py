"""One pre-norm transformer block, on the sentence used throughout the tutorial.

Four things happen here:

1. `layer_norm` standardises one position's vector across its d_model features,
   never across positions and never across the batch, as section 10 argues.
2. `feed_forward` is the position-wise network of section 10: two projections
   with a nonlinearity between them, widening to d_ff and coming back.
3. `block` wraps the two sublayers of sections 07 and 08 in the residual
   additions and the two norms, which is the whole of the transformer block.
4. The checks re-derive the numbers drawn in `layernorm_rows` and verify the
   properties the prose claims, so the figures, the text and the code cannot
   silently disagree.

Run:  uv run python topics/transformer_block.py
"""

import torch
import torch.nn.functional as F

WORDS = ["The", "cat", "chased", "the", "mouse"]

N = len(WORDS)  # 5 words
D_MODEL = 9  # section 07: h = 3 heads of width 3
H = 3
D_FF = 4 * D_MODEL  # 36, the usual choice

# Row (3) "chased" exactly as drawn in figures/build/layernorm_rows.svg.
# The nine values sum to 72, so mu = 8, and their squared deviations sum to
# 144, so sigma = 4.  Both are exact, which is why the figure prints no
# rounding.
ROW_3 = torch.tensor([9.0, 2.0, 13.0, 6.0, 15.0, 5.0, 10.0, 4.0, 8.0])
ROW_3_NORMALISED = torch.tensor(
    [0.25, -1.50, 1.25, -0.50, 1.75, -0.75, 0.50, -1.00, 0.00]
)


def layer_norm(X, gamma, beta, eps=1e-5):
    """
    X:     (..., d_model), the last axis is the one that gets normalised
    gamma: (d_model,) learned scale
    beta:  (d_model,) learned shift

    The statistics come from the last axis alone, so every position is
    normalised using only its own d_model numbers.  Nothing crosses a row,
    and nothing crosses the batch.
    """
    mu = X.mean(dim=-1, keepdim=True)
    var = X.var(dim=-1, unbiased=False, keepdim=True)
    return (X - mu) / torch.sqrt(var + eps) * gamma + beta


def feed_forward(X, W_1, b_1, W_2, b_2):
    """
    X:   (n, d_model)      W_1: (d_model, d_ff)   b_1: (d_ff,)
                           W_2: (d_ff, d_model)   b_2: (d_model,)

    Applied to each row on its own.  No row ever sees another row, which is
    exactly the contrast drawn in `mix_then_think`.
    """
    return F.relu(X @ W_1 + b_1) @ W_2 + b_2


def block(X, attn, ffn, ln_1, ln_2, mask=None):
    """
    One pre-norm block.  X is (n, d_model), and so is the result.

    Each sublayer reads a normalised copy and its output is *added* to the
    stream, so the stream itself is never overwritten and never renormalised.
    """
    Z = X + multi_head_attention(layer_norm(X, *ln_1), *attn, mask)
    Y = Z + feed_forward(layer_norm(Z, *ln_2), *ffn)
    return Y


def multi_head_attention(X, W_q, W_k, W_v, W_o, mask=None):
    """Section 07, unchanged, with section 08's mask added before the softmax."""
    Q, K, V = X @ W_q, X @ W_k, X @ W_v  # each (h, n, d_k)
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d_k**0.5  # (h, n, n)
    if mask is not None:
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    heads = weights @ V  # (h, n, d_v)
    concat = torch.cat([heads[m] for m in range(heads.shape[0])], dim=-1)
    return concat @ W_o  # (n, d_model)


def check_layernorm_figure_numbers():
    """Row (3) of `layernorm_rows`, re-derived rather than trusted."""
    mu = ROW_3.mean()
    sigma = ROW_3.std(unbiased=False)
    assert torch.allclose(mu, torch.tensor(8.0)), mu
    assert torch.allclose(sigma, torch.tensor(4.0)), sigma

    identity = torch.ones(D_MODEL), torch.zeros(D_MODEL)  # gamma = 1, beta = 0
    out = layer_norm(ROW_3, *identity)
    assert torch.allclose(out, ROW_3_NORMALISED, atol=1e-4), out

    print(f"row (3) before : {ROW_3.tolist()}")
    print(f"  mu = {mu:.2f}   sigma = {sigma:.2f}")
    print(f"row (3) after  : {[round(v, 2) for v in out.tolist()]}")
    print(f"  mu = {out.mean():.2f}   sigma = {out.std(unbiased=False):.2f}")


def check_normalisation_is_per_row():
    """Every row is normalised alone, so dropping the others changes nothing."""
    X = torch.randn(N, D_MODEL) * 5 + 3
    gamma, beta = torch.ones(D_MODEL), torch.zeros(D_MODEL)

    full = layer_norm(X, gamma, beta)
    alone = layer_norm(X[2:3], gamma, beta)

    assert torch.allclose(full[2:3], alone, atol=1e-6)
    print("\nrow (3) normalises identically with or without the other four rows")


def check_block_preserves_shape():
    """A block returns what it was given, which is what lets blocks stack."""
    d_k = D_MODEL // H
    X = torch.randn(N, D_MODEL)

    attn = (
        torch.randn(H, D_MODEL, d_k),  # W_q
        torch.randn(H, D_MODEL, d_k),  # W_k
        torch.randn(H, D_MODEL, d_k),  # W_v
        torch.randn(D_MODEL, D_MODEL),  # W_o
    )
    ffn = (
        torch.randn(D_MODEL, D_FF),
        torch.zeros(D_FF),
        torch.randn(D_FF, D_MODEL),
        torch.zeros(D_MODEL),
    )
    ln = (torch.ones(D_MODEL), torch.zeros(D_MODEL))

    Y = block(X, attn, ffn, ln, ln)
    assert Y.shape == X.shape == (N, D_MODEL), Y.shape

    # and because it is the same shape, it stacks
    for _ in range(5):
        Y = block(Y, attn, ffn, ln, ln)
    assert Y.shape == (N, D_MODEL)
    print(f"\none block: {tuple(X.shape)} -> {tuple(Y.shape)}, so six of them also fit")


def check_a_dead_sublayer_is_the_identity():
    """
    A sublayer with nothing to add contributes zero, and the block passes the
    stream through untouched.  This is why depth can never destroy what earlier
    blocks built.
    """
    X = torch.randn(N, D_MODEL)
    zero_ffn = (
        torch.zeros(D_MODEL, D_FF),
        torch.zeros(D_FF),
        torch.zeros(D_FF, D_MODEL),
        torch.zeros(D_MODEL),
    )
    ln = (torch.ones(D_MODEL), torch.zeros(D_MODEL))

    Z = X + feed_forward(layer_norm(X, *ln), *zero_ffn)
    assert torch.allclose(Z, X, atol=1e-6)
    print("a sublayer that outputs zero leaves the stream exactly as it found it")


def check_parameter_split():
    """Two thirds of a block is the feed-forward network."""
    attn_params = 4 * D_MODEL**2  # W_q, W_k, W_v summed over heads, plus W_o
    ffn_params = 2 * D_MODEL * D_FF  # = 8 * d_model^2 when d_ff = 4 * d_model
    assert ffn_params == 8 * D_MODEL**2
    total = attn_params + ffn_params
    print(
        f"\nattention {attn_params} weights, feed-forward {ffn_params} weights, "
        f"so the feed-forward is {ffn_params / total:.0%} of the block"
    )


if __name__ == "__main__":
    check_layernorm_figure_numbers()
    check_normalisation_is_per_row()
    check_block_preserves_shape()
    check_a_dead_sublayer_is_the_identity()
    check_parameter_split()
