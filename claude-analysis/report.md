# Inference-only tests on `torn-bigger-13` (general transpositions, $S_{25}$)

Run on 2026-11-25. Everything here uses the trained checkpoint as-is, with no retraining. Section 2.3 also uses one real training file and the real test file.
Code, raw results and the figure are in this folder; commands to reproduce are in §7.

## 1. Summary

1. **Baseline reproduced.** On the paper's real test file the model gets 2,000/2,000 words right, free-running. On freshly generated words free-running accuracy was 3,000/3,000. Teacher-forced word accuracy was 19,999/20,000, with 499,999/500,000 positions correct.
2. **Every test word is outside the training support.** The real training data touches at most 10 points (§2.3). All 2,000 words of the real test file touch 17 to 25 points (mean 21.6), and my fresh words behave the same way.
3. **But each individual chase in a natural test word is short.** In the real test file the longest chase made 9 hops and visited 9 distinct points. In my 20,000 fresh words, a chase visited at most 10 distinct points, and only 7 of 500,000 slots reached 10. So a test word is globally outside the support while each of its output slots is locally inside it. The paper-style accuracy therefore says nothing about slots with long chases.
4. **Planted-chase ("ladder") tests: accuracy is governed mainly by the number of hops $L$. The number of distinct points $s$ has a much smaller effect.**
   - Accuracy is essentially perfect up to $L\approx 12$ and falls steeply after $L\approx 15$.
   - At fixed $s=12$: 100% ($L=11$), 98.8% ($L=14$), 78.5% ($L=17$), 54.2% ($L=20$).
   - At $L=20$ the model is at about 50–56% whether the chase visits 4 points or 21 (1,000 words per cell in the fixed-hop sweep).
5. **So the hypothesis I proposed earlier, that accuracy collapses once a chase visits more distinct points than the training support ($s\ge11$), is not supported.** Chases with 11–14 distinct points (10–13 hops) were 99.8–100% correct. In the fixed-hop sweep (§3.3), crossing from $s\le10$ to $s\ge11$ costs a small, measurable amount, about 2 points at $L=16$ and 4 points at $L=20$ (standard errors about 1), compared with 30–35 points for going from $L=16$ to $L=20$.
6. **Unresolved: why hop count matters.** Either the 5-layer network cannot follow longer chains, or long chains are too rare in training, or both. The data is consistent with both (§3.3).
7. **Token embeddings are close to $S_{25}$-invariant** (§3.5), but one of my earlier predictions about their spectrum failed.
8. **The training support is $m=10$**, measured from the real training data (§2.3). The checkpoint config's `ACTUAL_GROUP_SIZE = 16` does not describe that data. In an earlier version of this report I wrongly took 16 from the config.

## 2. Setup and provenance

### 2.1 What was run
- The model's forward pass and greedy decoder were reimplemented in NumPy (`np_transformer.py`), mirroring `scaling-generator/transformer.py`, including the attention scaling by $n_{\text{embed}}^{-1/2}$ and the mask (inputs attend bidirectionally; outputs attend to all inputs and causally to earlier outputs).
- Weights come from `model.safetensors` (stored as bfloat16, upcast to float32).
- Free-running decoding uses `force_valid=True`, the repo's own metric. Teacher-forced evaluation takes the argmax over the whole vocabulary at each output slot, given the true preceding outputs.

### 2.2 Why I trust the reimplementation, and what is not checked
- The mask I build equals the mask buffer stored in the checkpoint.
- The cached incremental decoder agrees with a full forward pass to $1.7\times10^{-5}$ in the logits.
- It scores 100% on fresh words and 2,000/2,000 on the real test file. A faulty reimplementation would not.
- **Not done:** a direct comparison against the repo's PyTorch code. PyTorch could not be installed in my sandbox (the package host was blocked).

### 2.3 Test words and the training support
- The test set is not in this repo (the real one is in `Midataur/permutations`, see below). For most tests I generated words to match `fast-data-gen/src/main.rs` for the general type: 25 letters, each an independent uniform draw from the $25^2=625$ tokens (token $=25a+b$), with the permutation computed by swapping positions as in `get_permutation`.
- **Training support (measured).** The data is not in the latest commit of this repo. It is in `scaling-generator/data/torn_bigger/` at commit `5863de5` ("torn-bigger-13") of `Midataur/permutations`: 20 training files of 400,000 words each (8M words) plus a 2,000-word test file. I read one training file and the test file (`data_stats.py`, output in `data_stats.txt`).
  - Training words touch 7 to 10 distinct points (mean 9.93; 93% touch exactly 10) and never more than 10. Their permutations move at most 10 points. So $m=10$, as in the paper.
  - 10.0% of training letters are identity tokens, against 4.1% in the test file.
  - Test words touch 17 to 25 points (mean 21.6).
  - The stored permutations equal the ones I recompute from the words, in both files, so my reading of the token convention is right.
  - `trained-models/torn-bigger-13-config.py` sets `ACTUAL_GROUP_SIZE = 16`, which does not match this data. Earlier I switched to $m=16$ because you suggested trusting the config over the paper. That was wrong: the config value does not describe the data. I have not checked the other 19 training files, and I cannot prove the checkpoint was trained on exactly these files, but it scores 2,000/2,000 on this test file.

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
- At $L=20$ the accuracy is the same, within noise, for $s=4$ up to $s=12$ (400 words per cell). At $s=12$ it varies from 100% to 54% as $L$ goes from 11 to 20. So $L$ is the dominant variable. The larger fixed-hop sweep below finds a small additional dependence on $s$ across the support boundary.
- Where it fails, it fails softly: the median margin between the true logit and the best competitor goes from about 20 (easy) to about 0.6 at $L=20$. The model is close to indifferent rather than confidently wrong.
- At $L\ge19$ accuracy is around 50%, not near chance ($1/25$).
- Hop letters packed at the end of the word give about the same accuracy as spread-out ones (e.g. both 93.8% at $L=15$).
- Other output slots in the planted words stay at least 99% correct in every cell.

**Fixed-hop sweep across the support boundary.** The ladder cells confound $s$ and $L$, because a chase with $s$ distinct points needs at least $s-1$ hops. To test the boundary directly I fixed $L$ and varied $s$ from inside the training support ($s\le10$) to outside it ($s\ge11$). Script `run_beyond_support.py`, teacher-forced, 1,000 words per cell, seed 2027, result in `beyond_support_results.csv`.

| $L$ | $s$ | teacher-forced slot [95% CI] | median logit margin |
|---|---|---|---|
| 16 | 6 | 910/1000 (91.0%) [89.1, 92.6] | 11.6 |
| 16 | 9 | 895/1000 (89.5%) [87.4, 91.3] | 9.9 |
| 16 | 10 | 907/1000 (90.7%) [88.7, 92.3] | 9.7 |
| 16 | 11 | 896/1000 (89.6%) [87.6, 91.3] | 9.3 |
| 16 | 12 | 884/1000 (88.4%) [86.3, 90.2] | 8.9 |
| 16 | 17 | 876/1000 (87.6%) [85.4, 89.5] | 6.2 |
| 20 | 6 | 558/1000 (55.8%) [52.7, 58.9] | 1.2 |
| 20 | 9 | 529/1000 (52.9%) [49.8, 56.0] | 0.5 |
| 20 | 10 | 550/1000 (55.0%) [51.9, 58.1] | 0.7 |
| 20 | 11 | 508/1000 (50.8%) [47.7, 53.9] | 0.2 |
| 20 | 12 | 513/1000 (51.3%) [48.2, 54.4] | 0.2 |
| 20 | 17 | 499/1000 (49.9%) [46.8, 53.0] | 0.0 |
| 20 | 21 | 506/1000 (50.6%) [47.5, 53.7] | 0.0 |

- Hop count dominates. Going from $L=16$ to $L=20$ costs about 30–35 points at every $s$ (for example 88.4% to 51.3% at $s=12$).
- Pooling the cells inside the support ($s\le10$) and beyond it ($s\ge11$): at $L=16$ the accuracy is 90.4% inside and 88.5% beyond, a drop of 1.9 points (standard error 0.8). At $L=20$ it is 54.6% and 50.6%, a drop of 3.9 points (standard error 1.2).
- So there is a small but detectable cost to leaving the support, about 2 to 4 points. It looks more like a small step at $s=10\to11$ than a smooth slope: the cells inside the support are flat (52.9–55.8% at $L=20$) and so are those beyond it (49.9–51.3%). With 1,000 words per cell I can't rule out a gentle slope.
- The cost of leaving the support is about a tenth of the cost of adding hops.
- Caveat: at fixed $L$, larger $s$ also means a different word structure (fewer revisits of the same point), so this is not a pure change in support size.

**Relation to training frequency (from the real training data).** I counted, in one of the 20 real training files (400,000 words, §2.3), the output slots whose pointer chase makes at least $L$ hops, and scaled by 20 to the full 8M words. The tail is noisy because it rests on few observations.

| $L \ge$ | 12 | 13 | 15 | 17 | 19 |
|---|---|---|---|---|---|
| observed in file 1 | 2,310 | 483 | 18 | 1 | 0 |
| scaled to 8M words | 46,200 | 9,660 | 360 | 20 | 0 (upper 95% bound about 60) |
| model accuracy at this $L$ | 100% | 99.8% | 93.8% | 78.5% | 56.2% |

The accuracy cliff sits where the training data runs out of long chains: tens of thousands of slots with 12 or more hops, a few hundred at 15, about 20 at 17, and none seen at 19 or more. At $L=17$ there are only about 20 examples in the whole training set and the model still gets 78.5%. At $L=19$ to 20, where there are essentially none, it still gets about 55%. So it extrapolates partly beyond the chain lengths it saw, and degrades gradually instead of failing at the edge of its training data. That fits "long chains are rare in training". It equally fits "five layers can only follow so many hops", and this experiment cannot separate them. The numbers come from one file of 20.

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
- What the data does say is that the model's generalization beyond the training support is real at the level of whole words (all 2,000 real test words touch 17 or more points, against at most 10 in training), while each output slot only needs a short chase (at most 9 distinct points in the real test file). The ability to follow a chain falls off at $L\approx15$ to 20, which is where the training data runs out of long chains. Crossing the support boundary ($s=10\to11$) costs only 2 to 4 points. An accurate statement of what training "covers" is therefore about **chain length (hop count)** much more than about how many distinct points a word or a trajectory touches.
- In the spirit of the $\mathbb{Q}\subset\mathbb{R}$ analogy: the objects that need to be dense are trajectories (chases), not words. The training distribution is dense in short chains, wherever they sit among 25 points, and sparse in long ones. A density argument would need a metric in which "long chain" is far from "short chain".
- I can't say from this evidence whether the limit is architectural (depth) or statistical (rarity in training). Separating them needs experiments with new training runs (§6).

## 5. Corrections to my earlier messages
1. **Ladder hypothesis.** I framed the ladder test around a boundary at 11 distinct points, the first beyond the $m=10$ support. That boundary is correct (§2.3). There is no collapse at it (summary item 5).
2. **Word length.** My earlier closed-form tail estimates assumed 24 letters. The config uses 25 (`MAX_TRANS_NUMBER = 25`). Recomputed, the per-trajectory probability of at least 11 distinct points is $1.1\times10^{-6}$ (was $7.0\times10^{-7}$), and per word at most $2.7\times10^{-5}$ (was $1.7\times10^{-5}$). The simulation I ran earlier used 24 letters; the stratified run here uses 25 and saw none in 20,000 words, consistent with this.
3. **Training support.** I assumed $m=10$, then switched to $m=16$ on the config's say-so, and finally measured the real data: $m=10$ (§2.3). The config's `ACTUAL_GROUP_SIZE` does not describe the data.
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

Run from this folder (`claude-analysis/`). Scripts find the checkpoint at `../scaling-generator/trained-models/torn-bigger-13/model.safetensors`; set `TORN_MODEL` to override. Needs only `numpy` (plus `matplotlib` for figures); no PyTorch. To get the real data for `data_stats.py`, run `git cat-file blob 5863de5:scaling-generator/data/torn_bigger/<file>` in a clone of `Midataur/permutations` for `train_data1.csv`, `train_data1_perms.csv`, `test_data.csv` and `test_data_perms.csv`.

| file | purpose |
|---|---|
| `st_loader.py`, `np_transformer.py` | safetensors reader and NumPy forward/decoder |
| `permdata.py`, `evalutil.py` | word generation, pointer-chase statistics, planted words, teacher-forced evaluation |
| `sanity.py` | checks the decoder and pointer picture |
| `run_stratified.py N_FREE N_TF` then `analyse_strat.py` | baseline and stratified accuracy (I used `3000 20000`; about 7 minutes) |
| `run_ladder.py N_FREE N_TF` | ladder tests (I used `200 400`; about 5 minutes) |
| `run_beyond_support.py N_TF` | fixed-hop sweep across the $s=16$/$17$ boundary (I used `1000`; about 3 minutes) |
| `support_stats.py` | how many points fresh test words touch and move (seconds) |
| `data_stats.py DATA_DIR` | statistics of the real training file and test file, and the model on the real test file (§2.3; needs the data, see below) |
| `run_error_anatomy.py` | failure anatomy (§3.4) |
| `run_gram.py` | Gram spectrum (§3.5) |
| `make_figure.py` | draws `fig_accuracy_vs_hops.png` from `ladder_results.csv` |

Results included: `ladder_results.csv`, `beyond_support_results.csv`, `support_stats.txt`, `data_stats.txt`, `strat_summary.txt`, `gram_results.json`, `gram_spectrum.png`, `fig_accuracy_vs_hops.png`. The raw per-slot arrays from the stratified run (`strat_results.npz`, 2.3 MB) are not included; `run_stratified.py` regenerates them with the same seed.
