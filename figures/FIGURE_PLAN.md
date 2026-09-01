# Figure plan

One folder per figure under `figures/src/<name>/<name>.tex`, built to
`figures/build/<name>.svg` by `bash figures/src/build_figures.sh`.

Palette every figure must use:

```latex
\definecolor{c1}{HTML}{15173D}  % deep navy   - queries, structure, axes
\definecolor{c2}{HTML}{982598}  % magenta     - keys, the operative step
\definecolor{c3}{HTML}{E491C9}  % pink        - values, secondary emphasis
\definecolor{c4}{HTML}{F1E9E9}  % warm white  - panel fills, backgrounds
\definecolor{axiscolor}{HTML}{333333}
\definecolor{labelcolor}{HTML}{6B6472}
```

Role convention held across the whole tutorial, so the reader learns the colours
once: **query = c1 navy, key = c2 magenta, value = c3 pink, background panel =
c4**. Never swap these between figures.

---

## 01 Why Attention?

| name | shows |
|------|-------|
| `bottleneck_rnn` | An encoder RNN squeezing a whole sentence into one fixed vector, with the decoder starved on the other side. The visual argument for attention. |
| `lookup_vs_memorise` | Side by side: a model that must store everything in weights versus a model that keeps the context around and looks into it. |

## 02 The Database Analogy

| name | shows |
|------|-------|
| `hard_dictionary` | A classic key-value store: one query, exact match, one value returned. Everything else contributes zero. |
| `soft_dictionary` | The same store made differentiable: every key gets a match score, every value is returned weighted. The single figure the whole tutorial hangs on. |
| `qkv_roles` | Where $Q$, $K$, $V$ come from and what each one is responsible for, with the colour legend the rest of the tutorial reuses. |

## 03 Measuring Similarity

| name | shows |
|------|-------|
| `dot_product_geometry` | Two vectors, the projection of one onto the other, and the dot product as the signed length of that projection. |
| `score_landscape` | One query against several keys arranged around it, each with its raw score, showing which directions win. |

## 04 From Scores to Weights

| name | shows |
|------|-------|
| `softmax_bars` | Raw scores as bars, then the same scores after softmax as a distribution summing to one. |
| `temperature_sharpening` | The same score vector under three scalings: flat, calibrated, and collapsed onto one key. |
| `variance_growth` | Why the dot product's variance grows with $d_k$, and what dividing by $\sqrt{d_k}$ restores. |

## 05 Scaled Dot-Product Attention

| name | shows |
|------|-------|
| `attention_pipeline` | The full $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ pipeline as labelled matrix blocks, with every shape annotated. |
| `matrix_shapes` | A shape-tracking diagram: $(n \times d_k)$, $(n \times n)$, $(n \times d_v)$, so the reader can never lose the dimensions. |

## 06 Self-Attention

| name | shows |
|------|-------|
| `learned_projections` | One input matrix $X$ fanning out into $Q$, $K$, $V$ through three learned weight matrices. |
| `self_attention_sentence` | A short sentence with the attention weights of one chosen token drawn as arcs of varying thickness to every other token. |

## 07 Multi-Head Attention

| name | shows |
|------|-------|
| `head_split` | The embedding dimension split across heads, each head running its own small attention, then the concatenation and output projection. |
| `heads_ask_differently` | The same sentence, three heads, three different attention patterns (syntactic, positional, semantic). |

## 08 Masking and Causality

| name | shows |
|------|-------|
| `causal_mask` | The $n \times n$ score matrix with the upper triangle set to $-\infty$, and the same matrix after softmax. |
| `padding_mask` | A batch of unequal-length sequences and the mask that keeps padding tokens from receiving weight. |

## 09 Positional Information

| name | shows |
|------|-------|
| `permutation_equivariance` | The same tokens shuffled producing the same attention output, proving positions must be injected. |
| `sinusoidal_encoding` | The sinusoid bands across dimension and position, and how a dot product between two encodings falls off with distance. |
| `rope_rotation` | Rotary positions as a rotation of the query and key in 2-D planes, and why only relative offsets survive. |

## 10 The Transformer Block

| name | shows |
|------|-------|
| `residual_stream` | The block as a highway: attention writes into the stream, the feed-forward network writes into the stream, nothing overwrites it. |
| `prenorm_postnorm` | Where layer normalisation sits in each variant and what that does to gradient flow through a deep stack. |

## 11 The Cost of Looking Everywhere

| name | shows |
|------|-------|
| `quadratic_cost` | Score-matrix area growing with sequence length, with the memory number attached to a few realistic lengths. |
| `kv_cache` | Autoregressive decoding with and without the cache: what is recomputed and what is reused at each step. |
| `flash_tiling` | The score matrix processed in tiles that fit in fast memory, never materialising the full $n \times n$ matrix. |

## 12 Attention for Time Series

| name | shows |
|------|-------|
| `forecast_as_retrieval` | A long series where the model, asked to predict the next window, attends back to the earlier windows that rhyme with the present. |
| `token_is_timestamp` | Why a single timestamp is a poor token: almost no information, enormous sequence length, scores dominated by noise. |

## 13 Patching and Channel Independence

| name | shows |
|------|-------|
| `patching` | A raw series cut into overlapping patches, each patch flattened and projected into a token. Sequence length collapsing by the patch stride. |
| `channel_independence` | A multivariate series processed channel by channel through a shared backbone, versus one that mixes channels in the token. |

## 14 Building a Forecaster

| name | shows |
|------|-------|
| `forecaster_architecture` | The full model from windowed input to prediction head, every tensor shape labelled. |
| `train_val_windows` | The rolling-window split, showing what the model is allowed to see at train time versus evaluation time. |

## 15 Reading Attention Maps

| name | shows |
|------|-------|
| `attention_map_readout` | A real learned attention map beside the series it came from, with the periodicity the model latched onto marked. |
| `weights_are_not_explanations` | Two different weight patterns producing the same output, the counterexample to reading weights as explanations. |
