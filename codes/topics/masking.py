import torch
import torch.nn.functional as F


def masked_attention(raw_scores, true_lengths):
    """
    raw_scores:   (batch_size, seq_len, seq_len) prior to softmax
    true_lengths: (batch_size,) actual length of each sentence without padding
    """
    batch_size, seq_len, _ = raw_scores.shape

    # 1. Sequence (Causal) Mask: M
    # Creates a matrix with 0 on/below the diagonal, and -inf above it
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

    # 2. Padding Mask: P
    # Create an array of positions [0, 1, 2, ..., seq_len - 1]
    positions = torch.arange(seq_len).unsqueeze(0)

    # True where the position index is past the sentence's true length
    is_pad = positions >= true_lengths.unsqueeze(1)

    # Expand to (batch_size, 1, seq_len) to broadcast across all queries
    pad_mask = is_pad.unsqueeze(1)

    # 3. Apply Masks
    # Add causal mask (broadcasts identical triangle to all sequences)
    masked_scores = raw_scores + causal_mask

    # Overwrite padded columns with -inf
    masked_scores = masked_scores.masked_fill(pad_mask, float("-inf"))

    # 4. Softmax (-inf becomes 0.0)
    attention_weights = F.softmax(masked_scores, dim=-1)

    return attention_weights


if __name__ == "__main__":
    batch_size = 2
    max_seq_len = 5
    raw_scores = torch.randn(batch_size, max_seq_len, max_seq_len)
    true_lengths = torch.tensor([5, 3])
    print(masked_attention(raw_scores, true_lengths))
