"""Positional information, in PyTorch, on the sentence used throughout the tutorial.

Four things happen here, one per claim the section makes:

1. `check_permutation_equivariance` shows that attention without positions is
   blind to order: shuffle the rows of X and the output is the old output with
   its rows shuffled the same way, Attention(PX) == P Attention(X).
2. `check_positions_break_it` adds sinusoidal encodings and shows the equality
   fails, which is the whole point of adding them.
3. `check_shift_is_a_rotation` verifies that moving k positions along is a
   fixed rotation depending only on k, never on where we started.
4. `check_rope_depends_only_on_offset` verifies that a RoPE score depends on
   n - m alone, so the same offset gives the same score anywhere.

Run:  uv run python topics/positional.py
"""

import torch

WORDS = ["The", "cat", "chased", "the", "mouse"]


def attention(X, W_q, W_k, W_v):
    """One head, exactly as in section 06."""
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    scores = Q @ K.transpose(-2, -1)
    weights = torch.softmax(scores / Q.shape[-1] ** 0.5, dim=-1)
    return weights @ V


def sinusoidal(n, d_model):
    """The section 09 formula. Pair j turns at omega_j = 1 / 10000^(2j/d)."""
    p = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    j = torch.arange(d_model // 2, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (2 * j / d_model))
    angles = p * omega
    PE = torch.zeros(n, d_model)
    PE[:, 0::2] = torch.sin(angles)
    PE[:, 1::2] = torch.cos(angles)
    return PE


def check_permutation_equivariance():
    """Attention(PX) == P Attention(X). The claim the section opens with."""
    torch.manual_seed(0)
    n, d_model, d_k = len(WORDS), 8, 4
    X = torch.randn(n, d_model)
    W_q, W_k, W_v = (torch.randn(d_model, d_k) for _ in range(3))

    # swap cat (row 1) and mouse (row 4), the two words the section trades
    perm = torch.tensor([0, 4, 2, 3, 1])
    P = torch.eye(n)[perm]

    shuffled = attention(P @ X, W_q, W_k, W_v)
    moved = P @ attention(X, W_q, W_k, W_v)

    assert torch.allclose(shuffled, moved, atol=1e-5), (shuffled - moved).abs().max()
    print("without positions:")
    print("  Attention(PX) == P Attention(X), so the two sentences are one computation")
    return X, W_q, W_k, W_v, P


def check_positions_break_it(X, W_q, W_k, W_v, P):
    """Now shuffle the words but NOT the positions, which is what a different
    sentence actually is. Position 1 stays position 1 however the words move,
    because the position table is indexed by slot and not by token."""
    PE = sinusoidal(X.shape[0], X.shape[1])

    original = attention(X + PE, W_q, W_k, W_v)
    scrambled = attention((P @ X) + PE, W_q, W_k, W_v)

    # if order were still invisible, scrambled would just be original reordered
    gap = (scrambled - P @ original).abs().max().item()
    assert gap > 1e-3, f"positions changed nothing: {gap}"
    print("\nwith sinusoidal positions added:")
    print("  the words move between slots, the slots keep their own vectors")
    print(f"  the scrambled sentence is no longer the first one reordered, off by {gap:.3f}")


def check_shift_is_a_rotation():
    """PE(p + k) is PE(p) turned by an angle built only from k."""
    d_model, k = 16, 3
    PE = sinusoidal(64, d_model)
    j = torch.arange(d_model // 2, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (2 * j / d_model))
    b = k * omega                                  # the extra turn the shift adds

    for p in (0, 5, 17, 40):
        sin_p, cos_p = PE[p, 0::2], PE[p, 1::2]
        # the matrix from the aside, applied pair by pair
        sin_shift = torch.cos(b) * sin_p + torch.sin(b) * cos_p
        cos_shift = -torch.sin(b) * sin_p + torch.cos(b) * cos_p
        assert torch.allclose(sin_shift, PE[p + k, 0::2], atol=1e-5)
        assert torch.allclose(cos_shift, PE[p + k, 1::2], atol=1e-5)

    print(f"\nshifting by k={k} is one fixed rotation:")
    print("  the same matrix carried p to p+k from every starting point tried")


def rope(x, m, omega):
    """Turn each pair of coordinates of x by its own angle m * omega_j."""
    even, odd = x[0::2], x[1::2]
    a = m * omega
    out = torch.empty_like(x)
    out[0::2] = even * torch.cos(a) - odd * torch.sin(a)
    out[1::2] = even * torch.sin(a) + odd * torch.cos(a)
    return out


def check_rope_depends_only_on_offset():
    """<R_m q, R_n k> depends on n - m and on nothing else."""
    torch.manual_seed(1)
    d_k = 16
    j = torch.arange(d_k // 2, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (2 * j / d_k))
    q, k = torch.randn(d_k), torch.randn(d_k)

    pairs = [(1, 3), (5, 7), (9, 11), (100, 102)]     # every one an offset of 2
    scores = [torch.dot(rope(q, m, omega), rope(k, n, omega)).item() for m, n in pairs]

    for s in scores[1:]:
        assert abs(s - scores[0]) < 1e-4, scores

    print("\nRoPE scores for pairs two apart:")
    for (m, n), s in zip(pairs, scores):
        print(f"  positions {m:>3} and {n:>3}   {s: .4f}")
    print("  same offset, same score, wherever the pair sits in the sentence")


if __name__ == "__main__":
    state = check_permutation_equivariance()
    check_positions_break_it(*state)
    check_shift_is_a_rotation()
    check_rope_depends_only_on_offset()
