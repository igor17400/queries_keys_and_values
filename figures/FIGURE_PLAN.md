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

## 02 The Dictionary Analogy

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
| `next_token_targets` | The sentence *The cat chased the mouse* as five rectangles in a row, each labelled with its position **(1)** to **(5)**. Four curved arrows above the strip run from box $i$ to box $i+1$, the guess each position must make. Box **(1)** annotated "nothing precedes it, so it is never guessed at", box **(5)** annotated "nothing follows it, so it never guesses". Five words, four arrows. Resolves the two occurrences of *the* for the reader. Boxes c1 outline on c4 fill, arrows c2. |
| `causal_triangle` | Two 5x5 grids side by side for *The cat chased the mouse*, rows labelled with the querying word in c1, columns with the word being read in c2. Left, "every word reads every word": all 25 cells filled. Right, "every word reads only itself and what came before": the lower triangle including the diagonal stays filled, the upper triangle is struck through and empty. Row **(3)** *chased* framed in both, keeping (1)(2)(3) and losing (4)(5). Permission only: no $-\infty$, no numbers, no softmax, those belong to `causal_mask`. |
| `causal_mask` | Row **(3)** *chased* in three panels, carrying the numbers from `attention_pipeline` unchanged. (a) the scaled scores `-1.85 -0.34 -0.75 -1.44 0.86`; (b) the same row with the two forbidden entries replaced by $-\infty$; (c) after softmax, `0.12 0.53 0.35 0.00 0.00`, annotated "sums to 1". Shows the arithmetic on one row: the permission view is `causal_triangle`. |
| `padding_batch` | Two sentences of different lengths becoming one rectangular batch. Left, ragged: *The cat chased the mouse* as five boxes, *The mouse escaped* as three, with the two missing slots drawn as empty dashed outline so the ragged edge is obvious. An arrow labelled "pad to the longest". Right, a 2x5 block where sentence 2 carries three word boxes and two inert `[PAD]` boxes, annotated "one tensor, shape $2 \times 5$". Plumbing only: no scores, no mask, no $-\infty$, those belong to `padding_mask`. |
| `padding_mask` | A batch of unequal-length sequences and the mask that keeps padding tokens from receiving weight. |

## 09 Positional Information

| name | shows |
|------|-------|
| `two_lookups` | Three grids: the five word vectors (the two copies of *the* identical), the five position vectors (all different), and their sum, in which those two rows differ. Shows that position enters as a second lookup indexed by slot rather than by token, and that adding it is what separates repeated words. |
| `permutation_equivariance` | Two rows: the sentence, and the same sentence with *cat* and *mouse* swapped, each through the same attention block. Positions **(1)** to **(5)** are fixed in both, only the words move. The outputs hold the same five vectors with two rows traded, and the $\Pi$ entering on the left leaves on the right. Drawn: this is $\mathrm{Attention}(\Pi X) = \Pi\,\mathrm{Attention}(X)$. |
| `binary_to_sinusoid` | Four bit-rows of a binary counter over positions 0 to 15, each with the smooth wave of the same period drawn directly beneath it. The 7/8 boundary is marked, where every bit flips at once and the waves pass through unbothered. Carries the argument for sinusoids rather than illustrating the result. |
| `rope_rotation` | Rotary positions as a rotation of the query and key in 2-D planes, and why only relative offsets survive. |

## 10 The Transformer Block

| name | shows |
|------|-------|
| `blend_stays_inside` | The break that earns the feed-forward network. Two panels in the plane, axes labelled "two of the $d_v$ value dimensions" so the slice is honest. (a) Five points $v_1 \dots v_5$ labelled with the five words, their convex hull shaded c4 with a c1 outline; one c2 dot inside at $0.04v_1 + 0.18v_2 + 0.12v_3 + 0.06v_4 + 0.60v_5$, the answer row **(3)** *chased* came away with, sitting nearest *mouse*; a scatter of faint grey dots for other weight rows so the reachable set reads as a region rather than a point. Annotated "whatever the query, the answer is somewhere in here". (b) The same hull with the point now outside it, reached by an arrow labelled $\mathrm{FFN}$. Caption: a softmax row is positive and sums to one, so a head's answer is always a blend of value vectors and can never leave their hull; the feed-forward is what lets a position land somewhere new. Scope the claim to **one head, before $W_O$**: multi-head followed by $W_O$ is a sum of convex combinations and then a linear map, so the hull statement is exactly true only per head. |
| `mix_then_think` | The block in one contrast, and the most load-bearing figure of the section. Two panels over the same five row-boxes. (a) "attention: every row reads every row", five rows on the left and five on the right with all 25 thin c2 arrows drawn, so the panel reads as visibly dense. (b) "feed-forward: no row reads any other", the same five rows, each with its own two-layer network beside it at width $9 \to 36 \to 9$, and **zero arrows between rows**; $W_1$ and $W_2$ drawn once with a tie to all five rows marking them shared, in the thick c1 outline and dot that `attention_breakdown` uses for learned matrices. Caption: the two sublayers have exactly complementary jobs, attention the only place positions talk to each other, the feed-forward the only place a position is transformed. |
| `ffn_as_memory` | Optional, and worth drawing only if the key-value reading of the feed-forward stays a subsection rather than an aside. Laid out to deliberately rhyme with `soft_dictionary` from section 02: the same geometry and the same colours, but the rows of $W_1$ are the keys in c2, the rows of $W_2$ are the values in c3, and $\phi(xW_1)$ supplies the weights. Caption: a second lookup, this one written in the weights rather than read off the sentence. |
| `residual_stream` | The stream as a highway, for row **(3)** *chased* alone. A thick c1 line running three blocks' worth of the figure, so it reads as a highway rather than as one detour. Every sublayer hangs **off** the line: it branches, passes through a c2 box (`MultiHead`, then `FFN`), and returns to a $\oplus$ on the line. Nothing sits on the line itself. The width $d_{\mathrm{model}} = 9$ is stamped on the line and on both branch returns, annotated "the add is what forces every sublayer to give back the width it was given", which is what makes $W_O$'s shape in section 07 a requirement rather than a convention. No layer normalisation anywhere: the reader has not met it at this point in the section, and no figure may use a symbol before its definition. |
| `which_axis_normalises` | Optional, and about the axis only, never the arithmetic; the formula needs three lines of prose and no picture, but which axis is reduced is the thing readers get wrong. Three small $(\text{batch} \times \text{position} \times \text{feature})$ cubes with the reduction axis highlighted in c2: layer normalisation reducing over the features of one position, batch normalisation reducing across the batch, and the batch normalisation cube overlaid with the `[PAD]` slots from `padding_batch` to show that its statistics would be taken partly over filler. |
| `prenorm_postnorm` | Two columns, three blocks deep, in **identical layouts with one element displaced**, so the difference is visible before it is read. Left, post-norm: $\oplus$ and then $\mathrm{LN}$, with the norm box sitting **on** the main line. Right, pre-norm: the norm box on the branch, ahead of the sublayer, and the main line unbroken c1 from bottom to top. The same gradient path traced down both in c2: on the left it passes through all six norm boxes, on the right it touches none. The final $\mathrm{LN}$ before the output head is marked on the right column only. Caption: the difference is one box moved off the main line, and that is what keeps the identity path of `residual_stream` alive at depth. |
| `transformer_block` | The assembled block, and the diagram readers will scroll back to. One panel, every edge carrying its shape: $x$ $(5 \times 9)$ into $\mathrm{LN}$, into $\mathrm{MultiHead}$ with $h = 3$, into $\oplus$, giving $z$ $(5 \times 9)$; then $\mathrm{LN}$, $\mathrm{FFN}$ at $9 \to 36 \to 9$, $\oplus$, giving $y$ $(5 \times 9)$. Learned matrices in the thick-outline-and-dot convention. This is `residual_stream` completed, the same skeleton with the norms now placed, which is why the two figures both exist. |
| `full_stack` | The whole model, and the payoff for sections 01 to 10 together. Token ids into the embedding table of section 06, $\oplus$ the position vectors of section 09, into $L$ blocks drawn as three with an ellipsis, into the final $\mathrm{LN}$, into a linear map of shape $d_{\mathrm{model}} \times \lvert\mathcal{V}\rvert$, into a softmax, out to a bar chart over the vocabulary. Fed with the running sentence, and the distribution drawn is the one at position **(3)** *chased*, peaking on *the*, which is exactly the guess `next_token_targets` asked for in section 08. Closes the loop opened there and, further back, in section 01. |

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
