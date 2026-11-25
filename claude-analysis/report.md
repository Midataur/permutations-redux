# Inference-only tests on `torn-bigger-13` (general transpositions, $S_{25}$)

Run on 2026-11-25. Everything here uses the trained checkpoint as-is, with no retraining.
Code, raw results and the figure are in this folder; commands to reproduce are in §7.

## 1. Summary

1. **Baseline reproduced on freshly generated words.** Free-running accuracy was 3,000/3,000. Teacher-forced word accuracy was 19,999/20,000, with 499,999/500,000 positions correct.
2. **Every test word is globally out of distribution.** Across 20,000 words, the number of points moved by the output permutation was at least 12 (mean 21.0), whereas training words move at most $m$ points ($m$ is 10 in the paper; see §2.3 for a discrepancy).
3. **Natural test words contain almost no long chases.** Per output slot, the pointer chase (§3.1) visited at most 10 distinct points, and only 7 of 500,000 slots reached 10. The paper's accuracy therefore says nothing about slots with more than 10 distinct points.
4. **Planted-chase ("ladder") tests: accuracy tracks the number of hops $L$, not the number of distinct points $s$.**
   - Accuracy is essentially perfect up to $L\approx 12$ and falls steeply after $L\approx 15$.
   - At $L=20$ hops, teacher-forced accuracy is 53–57% for $s=4,6,8,10$ and 54% for $s=12$.
   - At fixed $s=12$: 100% ($L=11$), 98.8% ($L=14$), 78.5% ($L=17$), 54.2% ($L=20$).
5. **So the hypothesis I proposed earlier, that accuracy collapses once a trajectory has more distinct points than the training support, is not supported.** If $m=10$, chases with 11–14 distinct points were 99.8–100% correct. And the model fails with only 4 distinct points when the chase is long.
6. **Unresolved: why hop count matters.** Either the 5-layer network cannot follow longer chains, or long chains are too rare in training, or both. The data is consistent with both (§3.2).
7. **Token embeddings are close to $S_{25}$-invariant** (§3.4), but one of my earlier predictions about their spectrum failed.
8. **A question for you:** the checkpoint's config says `ACTUAL_GROUP_SIZE = 16`, but the paper says training used $S_{10}$. Which was used for this checkpoint? (§2.3)

## 2. Setup and provenance

### 2.1 What was run
- The model's forward pass and greedy decoder were reimplemented in NumPy (`np_transformer.py`), mirroring `scaling-generator/transformer.py`, including the attention scaling by $n_{\text{embed}}^{-1/2}$ and the mask (inputs attend bidirectionally; outputs attend to all inputs and causally to earlier outputs).
- Weights come from `model.safetensors` (stored as bfloat16, upcast to float32).
- Free-running decoding uses `force_valid=True`, the repo's own metric. Teacher-forced evaluation takes the argmax over the whole vocabulary at each output slot, given the true preceding outputs.

### 2.2 Why I trust the reimplementation, and what is not checked
- The mask I build equals the mask buffer stored in the checkpoint.
- The cached incremental decoder agrees with a full forward pass to $1.7\times10^{-5}$ in the logits.
- It scores 100% on fresh words. A faulty reimplementation would not.
- **Not done:** a direct comparison against the repo's PyTorch code. PyTorch could not be installed in my sandbox (the package host was blocked).

### 2.3 Test words, and a discrepancy about the training support
- The paper's test set is not in the repo. I generated words to match `fast-data-gen/src/main.rs` for the general type: 25 letters, each an independent uniform draw from the $25^2=625$ tokens (token $=25a+b$), with the permutation computed by swapping positions as in `get_permutation`.
- **Discrepancy:** `trained-models/torn-bigger-13-config.py` sets `MAX_GROUP_SIZE = 25` and `ACTUAL_GROUP_SIZE = 16`, with `RELABEL = True`. The paper says $S_{10}$. The training data is not in the repo and I could not tell from the files which support size was used. **I show both $m=10$ and $m=16$ wherever it matters.** The main conclusion in item 4 above does not depend on which is right, because the failures also occur at $s=4$, well inside either support.

## 3. Results

### 3.1 The pointer-chase picture (used throughout)
Reading a word right to left, output slot $k$ is
$$p_k = t_1\bigl(t_2(\cdots t_N(k))\bigr).$$
Start a "pointer" at $k$. At each letter $s_{a,b}$, if the pointer is at $a$ it moves to $b$ (and vice versa); otherwise it stays. A letter that moves it is a **hop**. For each slot I record $L$ (number of hops) and $s$ (number of distinct points visited, including the start). I check on every word that the pointer's final position equals the true $p_k$.

### 3.2 Stratified accuracy on natural words
Source: `strat_summary.txt` (20,000 words, 500,000 slots, teacher-forced).

| distinct points $s$ | slots | errors |
|---|---|---|
| 1–2 | 214,573 | 0 |
| 3 | 146,448 | 1 |
| 4 | 88,534 | 0 |
| 5 | 36,284 | 0 |
| 6 | 11,104 | 0 |
| 7 | 2,552 | 0 |
| 8 | 425 | 0 |
| 9 | 73 | 0 |
| 10 | 7 | 0 |

- The single error is at a slot with 2 hops and 3 distinct points, which is an easy case. One error in 500,000 is noise, and I draw no conclusion from it.
- The 95% interval for the rarest bin ($s=10$, 7 slots) is [0.65, 1.0], so it carries almost no information.
- Free-running: 3,000/3,000 (95% interval [0.9987, 1.0]).

### 3.3 Ladder (planted-chase) tests
**Design.** For one output slot I force the chase to make exactly $L$ hops and visit exactly $s$ distinct points. All other letters are random but avoid the pointer's current position, so they cannot move it. Hop letters sit at random word positions ("spread") or in the last $L$ letters ("contig"). Series:
- *Minimal ladders:* $L=s-1$ (every hop reaches a new point).
- *Control:* $L=20$ hops with $s=4,6,8,10$.
- *Fixed support:* $s=12$ with $L=14,17,20$ (plus $L=11$ from the ladder series).

**Figure:** `fig_accuracy_vs_hops.png` (top: accuracy vs $L$; bottom: expected number of training trajectories with at least $L$ hops under each candidate support).

| series | $L$ | $s$ | free-running word | teacher-forced slot [95% CI] | median logit margin |
|---|---|---|---|---|---|
| ladder | 3 | 4 | 200/200 | 400/400 | 20.8 |
| ladder | 7 | 8 | 200/200 | 400/400 | 21.1 |
| ladder | 10 | 11 | 200/200 | 400/400 | 20.4 |
| ladder | 11 | 12 | 200/200 | 400/400 | 19.6 |
| ladder | 12 | 13 | 200/200 | 400/400 | 18.4 |
| ladder | 13 | 14 | 199/200 | 399/400 [98.6, 100] | 16.2 |
| ladder | 15 | 16 | 192/200 | 375/400 (93.8%) [90.9, 95.7] | 10.2 |
| ladder | 19 | 20 | 116/200 | 225/400 (56.2%) [51.4, 61.0] | 0.7 |
| ladder | 23 | 24 | 101/200 | 174/400 (43.5%) [38.7, 48.4] | −0.4 |
| control | 20 | 4 | 96/200 | 221/400 (55.2%) [50.4, 60.0] | 0.8 |
| control | 20 | 6 | 104/200 | 224/400 (56.0%) [51.1, 60.8] | 1.8 |
| control | 20 | 8 | 118/200 | 228/400 (57.0%) [52.1, 61.8] | 1.2 |
| control | 20 | 10 | 110/200 | 213/400 (53.2%) [48.4, 58.1] | 0.5 |
| fixed $s=12$ | 14 | 12 | 197/200 | 395/400 (98.8%) [97.1, 99.5] | 15.0 |
| fixed $s=12$ | 17 | 12 | 161/200 | 314/400 (78.5%) [74.2, 82.2] | 5.0 |
| fixed $s=12$ | 20 | 12 | 119/200 | 217/400 (54.2%) [49.4, 59.1] | 0.6 |
| contig | 11 | 12 | 198/200 | 398/400 (99.5%) | 12.6 |
| contig | 15 | 16 | 198/200 | 375/400 (93.8%) | 7.2 |

The full table (24 cells, including the remaining contig cells) is in `ladder_results.csv`.

**What it shows.**
- At $L=20$ the accuracy is the same, within noise, for $s=4$ up to $s=12$. At $s=12$ it varies from 100% to 54% as $L$ goes from 11 to 20. So $L$, not $s$, drives the failure.
- Where it fails, it fails softly: the median margin between the true logit and the best competitor goes from about 20 (easy) to about 0.6 at $L=20$. The model is close to indifferent rather than confidently wrong.
- At $L\ge19$ accuracy is around 50%, not near chance ($1/25$).
- Hop letters packed at the end of the word give about the same accuracy as spread-out ones (e.g. both 93.8% at $L=15$).
- Other output slots in the planted words stay at least 99% correct in every cell.

**Relation to training frequency (suggestive only).** Assuming 8M training words of 25 uniform letters on $m$ points (the dataset size is from the paper's hyperparameter table; the rest is an assumption), the expected number of training trajectories with at least $L$ hops is:

| $L \ge$ | 12 | 13 | 15 | 17 | 19 |
|---|---|---|---|---|---|
| $m=10$ | 46,300 | 9,790 | 280 | 4.3 | 0.03 |
| $m=16$ | 1,020 | 132 | 1.4 | 0.008 | $2\times10^{-5}$ |

The accuracy cliff (about $L=15$–19) sits where the expected training count drops from hundreds to about zero. That fits a "long chains are rare in training" explanation. It equally fits a "five layers can only follow so many hops" explanation, and this experiment cannot separate them.

**Limitations of the design.** At $L=20$, 80% of the word's letters belong to one chain, so planted words differ from training words in other ways too (§4). The cells use 200 free-running and 400 teacher-forced words, so intervals are about ±5 percentage points. Teacher forcing lets the model see correct earlier outputs, which may help it.

### 3.4 Failure anatomy (exploratory, weak evidence)
`run_error_anatomy.py` compares each wrong teacher-forced prediction with the pointer's positions after $0,1,\dots,L$ hops.
- For the minimal ladders this is uninformative: the chase path covers 20–24 of 25 points, so landing on it is nearly guaranteed.
- For the few-point control ($s=4$) 71% of failures land on the chase path (about 16% expected by chance), which suggests failures are often intermediate states of the chase. But with only 4 points many hop counts map to the same point, which biases the "hops short of the end" measure downward.
- For $s=12$, $L=20$: 79% land on the path (about 46% by chance), and the shortfall is spread widely (median 6 hops), not concentrated just short of the end.
- So there is no clean "stops after $r$ hops" signature. I would not build on this.

### 3.5 Gram spectrum of the token embeddings
Tokens $s_{a,b}$ ($a<b$) are indexed by 2-element subsets of $\{1,\dots,25\}$, and $S_{25}$ acts on them. The 300-dimensional permutation module splits as
$$M^{(23,2)}=S^{(25)}\oplus S^{(24,1)}\oplus S^{(23,2)},\qquad 300=1+24+275.$$
An $S_{25}$-invariant Gram matrix would be a combination of the three isotypic projectors (the Bose–Mesner algebra of the Johnson scheme $J(25,2)$). `run_gram.py` measures this, with a null made by randomly relabelling the 300 tokens.

| quantity | trained | null (random relabelling) |
|---|---|---|
| fraction of Gram energy outside the three diagonal blocks | 0.0075 | 0.13 |
| off-diagonal energy not explained by the 3 class means | 0.021 | 1.00 |
| within-block relative eigenvalue spread (V0, V1, V2) | 0, 0.083, 1.01 | 0, 0.80, 2.64 |

- The embedding Gram is very close to invariant.
- About 82% of the symmetrised embedding energy lies in the additive part ($V_0\oplus V_1$, i.e. roughly $u_a+u_b$); regressing $E[s_{ab}]$ on $u_a+u_b$ gives $R^2=0.71$.
- **A prediction I made earlier did not hold cleanly:** I expected three eigenvalue clusters. $V_1$ is nearly one eigenvalue (28.8, spread 8%), but the 275-dimensional block ($V_2$, mean eigenvalue 0.54) has a relative spread of about 1.0, so it does not collapse to a single value. It is tighter than the null but not clean.
- $E[s_{ab}]$ and $E[s_{ba}]$ have cosine similarity about 0.74 (min 0.67). The paper described them as equal. They are related but not identical.
- Only the general-transposition model was analysed.

## 4. What this means for the density idea

- The algebraic reason the target is local (Moore's presentation: every relation involves at most 4 points, and the pointer chase needs one point at a time) stands. The experiments above don't test it directly.
- What the data does say is that the model's generalization in the number of points is real, but its ability to follow a chain has a length limit well below the 25 letters available. An accurate statement of what training "covers" is therefore about **chain length (hop count)**, not about how many distinct points a trajectory touches.
- In the spirit of the $\mathbb{Q}\subset\mathbb{R}$ analogy: the training distribution is dense in short chains, wherever they sit among 25 points, and sparse in long ones. A density argument would need a metric in which "long chain" is far from "short chain".
- I can't say from this evidence whether the limit is architectural (depth) or statistical (rarity in training). Separating them needs experiments with new training runs (§6).

## 5. Corrections to my earlier messages
1. **Ladder hypothesis.** I framed the ladder test as "collapse at 11 distinct points means bounded-support patterns". There is no collapse at 11; the data refute that version.
2. **Word length.** My earlier closed-form tail estimates assumed 24 letters. The config uses 25 (`MAX_TRANS_NUMBER = 25`). Recomputed, the per-trajectory probability of at least 11 distinct points is $1.1\times10^{-6}$ (was $7.0\times10^{-7}$), and per word at most $2.7\times10^{-5}$ (was $1.7\times10^{-5}$). The simulation I ran earlier used 24 letters; the stratified run here uses 25 and saw none in 20,000 words, consistent with this.
3. **Training support.** I assumed $m=10$ throughout; the config says 16 (§2.3).
4. **Gram spectrum.** See §3.5.
5. **Untested:** my earlier suggestion that $m=4$ or $5$ might already suffice has not been tested (it needs retraining).

## 6. Not done, and suggested next steps

**Inference-only tests not yet run:** irrelevance interventions (replace letters that cannot touch the pointer and check $p_k$ is unchanged); gap/alignment variations beyond spread vs contiguous; linear probes for the pointer's position after $r$ hops at each layer; attention patterns and activation patching; weight-level equivariance; the elementary-transposition model (my NumPy model is specific to the general vocabulary layout).

**Needs training (not possible in my environment):** the support-size sweep; models of different depth; training with long-chain-enriched data.

Given §3.3, I would prioritise:
1. **Layer-wise probes** for the pointer position after $r$ hops. If intermediate positions are decodable at specific layers, the cliff may reflect the number of layers.
2. **Irrelevance interventions** at moderate $L$ (say 8–12), to test whether outputs depend only on the chase letters.
3. **Training with long-chain-enriched data** (or a deeper model). This is the cleanest way to separate "rare in training" from "too deep for the architecture": if the cliff moves with the data, it was coverage.

## 7. Reproducing

Run from this folder (`claude-analysis/`). Scripts find the checkpoint at `../scaling-generator/trained-models/torn-bigger-13/model.safetensors`; set `TORN_MODEL` to override. Needs only `numpy` (plus `matplotlib` for figures); no PyTorch.

| file | purpose |
|---|---|
| `st_loader.py`, `np_transformer.py` | safetensors reader and NumPy forward/decoder |
| `permdata.py`, `evalutil.py` | word generation, pointer-chase statistics, planted words, teacher-forced evaluation |
| `sanity.py` | checks the decoder and pointer picture |
| `run_stratified.py N_FREE N_TF` then `analyse_strat.py` | baseline and stratified accuracy (I used `3000 20000`; about 7 minutes) |
| `run_ladder.py N_FREE N_TF` | ladder tests (I used `200 400`; about 5 minutes) |
| `run_error_anatomy.py` | failure anatomy (§3.4) |
| `run_gram.py` | Gram spectrum (§3.5) |
| `make_figure.py` | draws `fig_accuracy_vs_hops.png` from `ladder_results.csv` |

Results included: `ladder_results.csv`, `strat_summary.txt`, `gram_results.json`, `gram_spectrum.png`, `fig_accuracy_vs_hops.png`. The raw per-slot arrays from the stratified run (`strat_results.npz`, 2.3 MB) are not included; `run_stratified.py` regenerates them with the same seed.
