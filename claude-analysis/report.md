# Inference-only tests on `torn-bigger-13` (general transpositions, $S_{25}$)

Run on 2026-11-25. Everything here uses the trained checkpoints as-is, with no retraining. Section 2.3 also uses one real training file and the real test file.
Code, raw results and the figures are in this folder; commands to reproduce are in §7.

**What is new in this version (the second version of this report).** New material is marked **[NEW]**. Text with no marker is unchanged from the first version, except where marked **[UPDATED]**.

- **§3.6 [NEW]:** Test 1, layer-wise linear probes (general-transposition model, `torn-bigger-13`).
- **§3.7 [NEW]:** Test 2, irrelevance interventions (same model).
- **§3.8 [NEW]:** Test 6a, the elementary-transposition model (`elem-partitioned-long-boosted-1`) ported to NumPy and tested for width and hop count, plus the position-45 question.
- **§1 [UPDATED]:** summary items 9–12 are new, and item 6 now points to them.
- **§4, §6, §7 [UPDATED]:** one new paragraph in §4; §6 rewritten (tests 1, 2 and 6a are now done); new scripts and result files in §7.

## 1. Summary

1. **Baseline reproduced.** On the paper's real test file the model gets 2,000/2,000 words right, free-running. On freshly generated words free-running accuracy was 3,000/3,000. Teacher-forced word accuracy was 19,999/20,000, with 499,999/500,000 positions correct.
2. **Every test word is outside the training support.** The real training data touches at most 10 points (§2.3). All 2,000 words of the real test file touch 17 to 25 points (mean 21.6), and my fresh words behave the same way.
3. **But each individual chase in a natural test word is short.** In the real test file the longest chase made 9 hops and visited 9 distinct points. In my 20,000 fresh words, a chase visited at most 10 distinct points, and only 7 of 500,000 slots reached 10. So a test word is globally outside the support while each of its output slots is locally inside it. The paper-style accuracy therefore says nothing about slots with long chases.
4. **Planted-chase ("ladder") tests: accuracy is governed mainly by the number of hops $L$. The number of distinct points $s$ has a much smaller effect.**
   - Accuracy is essentially perfect up to $L\approx 12$ and falls steeply after $L\approx 15$.
   - At fixed $s=12$: 100% ($L=11$), 98.8% ($L=14$), 78.5% ($L=17$), 54.2% ($L=20$).
   - At $L=20$ the model is at about 50–56% whether the chase visits 4 points or 21 (1,000 words per cell in the fixed-hop sweep).
5. **So the hypothesis I proposed earlier, that accuracy collapses once a chase visits more distinct points than the training support ($s\ge11$), is not supported.** Chases with 11–14 distinct points (10–13 hops) were 99.8–100% correct. In the fixed-hop sweep (§3.3), crossing from $s\le10$ to $s\ge11$ costs a small, measurable amount, about 2 points at $L=16$ and 4 points at $L=20$ (standard errors about 1), compared with 30–35 points for going from $L=16$ to $L=20$.
6. **Unresolved: why hop count matters.** Either the 5-layer network cannot follow longer chains, or long chains are too rare in training, or both. The data is consistent with both (§3.3). **[UPDATED]** The new tests narrow this but do not settle it; see item 12.
7. **Token embeddings are close to $S_{25}$-invariant** (§3.5), but one of my earlier predictions about their spectrum failed.
8. **The training support is $m=10$**, measured from the real training data (§2.3). The checkpoint config's `ACTUAL_GROUP_SIZE = 16` does not describe that data. In an earlier version of this report I wrongly took 16 from the config.
9. **[NEW] Layer-wise probes (§3.6).** At the position that predicts $p_k$, the pointer's location after $r$ hops is linearly decodable for the first few hops only. For $r=1$ a probe reaches 97–99% at layer 3, for $r=4$ about 60% at layers 4–5, and for $8\le r\le L-2$ it is at most 13% (chance 4%) in every layer. That holds even for $L=8$ and $L=12$, where the model is 100% correct. The answer itself appears only after the last block. So the model does not keep a step-by-step record of the chain at the query position, and there is no staircase of one hop per layer. This does not exclude composition at the input positions, which I did not probe.
10. **[NEW] Irrelevance interventions (§3.7).** For $L\le 12$ and for natural words the prediction depends on the hop letters and on nothing else. Replacing every letter that cannot touch the pointer leaves the prediction unchanged in 98.1–100% of 800 words per set, and replacing one hop letter changes it in about 96% (planted chases). At $L=16$ and $L=20$ the same replacements change 16% and 45% of predictions, almost all of them near-ties, and accuracy does not fall (87.1% to 88.0%, and 48.5% to 50.4%). At $L=20$, replacing a hop letter that is read last, i.e. one of the chain's final hops, changes the prediction only about 60% of the time, against 95–98% for hop letters read first.
11. **[NEW] Elementary-transposition model (§3.8).** My NumPy port reproduces the model: 31,996 of 32,000 slots correct on the real test file (99.9875%, teacher-forced) and 500/500 words free-running. At fixed width 8 (the pointer stays inside 8 adjacent positions) accuracy is 99.8–100% from 12 up to 80 hops. There is no hop-count cliff, and its training data contains chases of up to 78 hops (about 0.1% of slots have 56 or more). Accuracy falls only when the chase spans more positions than any training chase did (width at least 12): 95.6% at width 16 with 24 hops, 90.0% at width 16 with 40 hops. On 3,000 fresh words, 7 of the 8 errors are at width 11 or more, which holds 590 of the 48,000 slots.
12. **[NEW] What this does to item 6.** The elementary model has the same architecture family (5 blocks, 6 heads; my NumPy code loads both) and follows chases of 40–80 hops when its training data contains chases that long. The general-transposition model's cliff at 15–20 hops sits where its own training data ends (§3.3). That favours "rare in training" over "five layers cannot do it", but it is not a proof: the two tasks differ (§3.8), and I have not trained the general model on longer chains.

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

### 3.6 Layer-wise probes (Test 1) [NEW]

**Question.** Does the model carry the pointer's position after $r$ hops at the position that predicts $p_k$, and in which layer does each one appear? If $x_r$ showed up at layer $\approx r$ and the staircase ran out near $L\approx 15$, the five layers would be the bottleneck. If many hops showed up at once, the model would compose in parallel and the cliff would need another explanation.

**Design.** Planted chases with $L$ hops and $s$ distinct points, 4,000 words per design, designs $(L,s)=(8,9),(12,13),(16,17),(20,21),(20,6)$. For each word I ran the model with teacher forcing and took the residual-stream vector at position $25+k$ (the position whose output is $p_k$), after the embeddings (layer 0) and after each of the five blocks (layers 1–5, before the final LayerNorm). Let $x_r$ be the pointer's position after $r$ hops, so $x_0=k$ and $x_L=p_k$. For every layer and every $r$ I fitted a ridge-regression probe from the vector to the 25-way label $x_r$ on 60% of the words, chose the ridge strength on 20%, and report accuracy on the remaining 20% (800 words, standard error about 1.8 points at 50%). Chance is 4%. A control with randomly shuffled training labels gave 3.7–4.6% at layers 0 and 5 in every design (`probe_controls.json`). Script `probe_layers.py`.

![Probe accuracy for the pointer position after r hops, by layer, at the query position. Top: five designs (the colour scale is accuracy, chance 0.04). Bottom: for the designs where the model sometimes fails, probes at layers 2 and 5 split by whether the model got the slot right.](fig_probe_layers.png)

Probe accuracy in percent, $L=12$, $s=13$ (the model is correct on all 800 test words here, so there is no split by outcome):

| layer | $x_1$ | $x_2$ | $x_3$ | $x_4$ | $x_5$ | $x_6$ | $x_8$ | $x_{12}$ (the answer) |
|---|---|---|---|---|---|---|---|---|
| 0 (embeddings) | 4 | 4 | 5 | 5 | 4 | 4 | 4 | 4 |
| 1 | 42 | 18 | 13 | 9 | 4 | 5 | 5 | 20 |
| 2 | 84 | 38 | 12 | 7 | 6 | 6 | 3 | 18 |
| 3 | 98 | 78 | 65 | 19 | 6 | 6 | 5 | 18 |
| 4 | 91 | 74 | 70 | 60 | 38 | 24 | 5 | 18 |
| 5 | 85 | 71 | 69 | 58 | 36 | 24 | 6 | 100 |

**What it shows** (the first four designs, which are minimal ladders: every $x_r$ is a different point).
- $x_0=k$ is decodable at 100% in every layer. It is given by the query's position, so it says nothing about mechanism.
- **Early hops appear in order, about one layer per hop.** $x_1$ is at 38–47% after layer 1, 84–88% after layer 2 and 97–99% after layer 3. $x_2$ peaks at 75–86%, $x_3$ at 67–79% and $x_4$ at 47–66%, each a layer or so later than the one before.
- **The staircase stops after about five hops.** $x_5$ and $x_6$ reach at most 51% and 37%, and $x_r$ for $8\le r\le L-2$ stays at 3–13% in every layer, for every design with $L\ge12$. (The state one hop before the answer, $x_{L-1}$, reaches 11% at $L=16$ and 16% at $L=20$ at layer 3, a little above chance.) At $L=8$, where the model is 100% correct, $x_5,x_6,x_7$ are decodable at only 21%, 10% and 9% in the last layer. So the model gets the right answer for 8 and 12 hops while the later intermediate positions are barely present at the query position.
- **The answer appears in the last block.** At layers 1–4 the probe for $x_L$ gives 14–20%, which is above chance and which I do not explain. At layer 5 it gives 100% for $L=8,12$, 92% for $L=16$ and 61% for $L=20$, close to the model's own accuracy (100%, 85.5%, 48.6% at the planted slot).
- **Correct versus wrong (bottom panels).** At layer 2 the probes behave almost identically on slots the model gets right and wrong. At layer 5 the wrong cases show higher, not lower, decodability for the early hops (for example $x_2$ at $L=16$: 89% on wrong slots against 76% on right ones). I have no explanation, and the split is exploratory.
- **The $(20,6)$ panel is not comparable.** With only 6 points, many $x_r$ coincide with $x_0$ (which is free) or with one another, so even the embeddings decode $x_2$ at 78%. I would not read it.

**What this does and does not say.**
- A purely sequential chase, one hop per layer with the state carried at the query position, cannot give correct answers for 8–12 hops with five blocks, and the probes agree: there is no sign of it past about 5 hops. So "the staircase runs out around $L\approx15$" is not what happens. The first few hops are resolved in layers 1–4 and the rest is not visible.
- **How the model gets the later hops is not visible here.** Candidates are composition at the input-word positions (for example combining blocks of letters in parallel), a nonlinear encoding, or attention reading the letters directly. The probes cannot distinguish these.
- Linear decodability is not use, and a failed probe does not prove absence.
- Not done: probes at the input-word positions, and activation patching.

### 3.7 Irrelevance interventions (Test 2) [NEW]

**Question.** Does the model compute only along the chase, or do letters that cannot touch the pointer influence the answer? If they do, failures at long $L$ could partly be distraction.

**Design.** For an output slot $k$ a letter is *irrelevant* if, at its turn in the right-to-left chase, it does not contain the pointer's current position (so it cannot change $p_k$). I replace irrelevant letters by other letters that also avoid the pointer, so the true $p_k$ is unchanged by construction (checked on every word), and ask whether the model's prediction at slot $k$ changes. Conditions: all irrelevant letters replaced (`full`), each with probability 1/2 (`half`), or all replaced by identity tokens (`identity`). As a positive control, one random hop letter is replaced by an identity token, which changes the true $p_k$ (`delete1`). Word sets are planted chases with $L=8,12,16,20$ and natural random words (random slot), 800 each. The output prefix fed to the model is always the original word's true outputs, so only the input letters differ. Script `run_irrelevance.py`.

Percentage of words whose prediction is unchanged (accuracy at the slot, before and after, for `full`):

| word set | accuracy before → after (`full`) | `full` | `half` | `identity` | `delete1` (control) |
|---|---|---|---|---|---|
| planted $L=8$ | 100 → 100 | 100.0 | 100.0 | 99.6 | 3.6 |
| planted $L=12$ | 100 → 100 | 100.0 | 99.8 | 100.0 | 4.2 |
| planted $L=16$ | 87.1 → 88.0 | 84.2 | 86.5 | 85.5 | 4.9 |
| planted $L=20$ | 48.5 → 50.4 | 54.5 | 67.4 | 60.4 | 14.5 |
| natural words | 100 → 100 | 100.0 | 100.0 | 98.1 | 14.6 |

The `delete1` figure for natural words, 14.6%, is the share of slots with no hop at all (nothing to delete), so it is not a failure of the control.

**Confident versus near-tied predictions.** Splitting by the model's baseline logit margin (confident: margin above 5) shows where the changes at long $L$ come from:

| set | prediction unchanged, confident (n) | prediction unchanged, near-tied (n) |
|---|---|---|
| $L=16$, `full` | 96.4% (501) | 63.9% (299) |
| $L=20$, `full` | 99.0% (104) | 47.8% (696) |

![Influence of single letters. Each point is the share of cases in which replacing one letter changes the model's prediction, by the letter's reading order. Blue: the letter is a hop letter (replaced by an identity token). Orange: the letter is irrelevant (replaced by a random pointer-avoiding letter). 300 words per panel.](fig_influence_map.png)

**What it shows.**
- **Locality holds at $L\le12$ and on natural words.** Replacing every irrelevant letter (13 to 23 letters per word) leaves the prediction unchanged in 99.8–100% of cases under `full` and `half`, and in 98.1–100% under `identity`. Every hop letter matters: replacing a single one changes the prediction in 94.6% of the 3,600 hop letters tested at $L=12$ (0.2% for irrelevant letters), with no strong dependence on where the letter sits in the word.
- **At $L=16$ and $L=20$ the predictions do move, but mostly where the model was undecided.** Confident predictions are 96–99% stable, near-tied ones 48–64%. Accuracy does not fall (87.1% to 88.0%, 48.5% to 50.4%; with 800 words the standard error is about 1.2 to 1.8 points), so I see no systematic distraction. The changes look like a near-tie being re-rolled by any perturbation.
- **At $L=20$ the model partly ignores the chain's last hops.** Replacing a hop letter that is read last (the first letters of the word) changes the prediction only about 60% of the time, and the share rises steadily to 95–98% for letters read first (right panel). Among the 104 confident $L=20$ words, replacing one random hop leaves the prediction unchanged in 34%. This fits a model that sometimes stops following a long chain before the end. It does not prove that, and my earlier failure anatomy (§3.4) found no clean "stops after $r$ hops" signature.

**Caveats.** Planted words at $L=16$ and $20$ are unlike training words anyway (§3.3). Replacing irrelevant letters with identity tokens makes natural words almost all identity, which is also unusual. Because the output prefix is the *original* one, other slots' prefix tokens may disagree with the changed input. The model's output at slot $k$ ignored this at $L\le12$, which, if anything, strengthens the locality result.

### 3.8 Elementary-transposition model (Test 6a, and the position-45 question) [NEW]

**The model.** `elem-partitioned-long-boosted-1` (checkpoint in this repo; training and test data in `Midataur/permutations` at commit `fdefb8c`). A word has 120 letters, each one of 16 tokens. Token $t\ge1$ swaps positions $t-1$ and $t$ and token 0 is the identity, so the group is $S_{16}$ and the model emits 16 output slots. The vocabulary has 34 tokens (16 letter tokens, 16 output tokens, NULL, START) and the context is 136. The pointer chase is the same, but every hop moves the pointer by $\pm1$. Instead of the number of distinct points I use the **width** $w$ of a chase: the number of adjacent positions it spans (largest minus smallest position visited, plus 1).

**Port.** `np_transformer.py` now takes `max_group`, `input_len` and `num_trans`, and loads this model as `NPTransformer(path, max_group=16, input_len=120, num_trans=16)`. The mask I build equals the stored one. As before I could not compare with the repo's PyTorch code; the check is the real test file below.

**Real data** (first 100,000 rows of `train_data1.csv`, i.e. 1.6M slots, and the 2,000-word test file; `elem_stats.py`, output `elem_stats.txt`).
- The stored permutations equal the ones I recompute (token $t$ swaps positions $t-1$ and $t$).
- Training chases: hops per slot have mean 14.7, quantiles (5%, 50%, 95%, 99%, 99.9%) of 0, 16, 38, 47, 56, and a maximum of 78. Widths are at most 11 (no chase spans 12 or more positions); each training row uses at most 10 distinct tokens, from a window of up to 16.
- Test file: hops mean 14.1 and maximum 29; width mean 5.6 and maximum 15; 0.53% of slots have width 12 or more.
- **Model on the real test file:** 31,996 of 32,000 slots correct teacher-forced (99.9875%), 1,996 of 2,000 words, and 500 of 500 words free-running (first 500 test words). The four wrong slots (hops, width): (19, 14), (5, 4), (12, 13), (17, 13). Three of the four are at width 13 or more.

**Planted cells** (`run_elem_cells.py`; 500 words per cell, teacher-forced at the planted slot, seed 4242; `elem_cells_results.csv`). The planted chase makes exactly $L$ hops and spans exactly $w$ positions; other letters avoid the pointer.

Teacher-forced accuracy at the planted slot (500 words per cell). Series A fixes the width at 8 and varies the hops; series B and C fix the hops at 24 and 40 and vary the width.

| A: width $w=8$, hops $L$ | 12 | 24 | 40 | 56 | 80 |
|---|---|---|---|---|---|
| correct (of 500) | 500 | 500 | 500 | 500 | 499 |
| median logit margin | 22.9 | 22.8 | 21.9 | 20.6 | 16.7 |

| B: $L=24$ hops, width $w$ | 4 | 8 | 11 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|
| correct (of 500) | 500 | 500 | 500 | 500 | 496 | 478 |
| median logit margin | 23.1 | 23.1 | 22.5 | 21.8 | 18.2 | 12.1 |

| C: $L=40$ hops, width $w$ | 8 | 11 | 12 | 16 |
|---|---|---|---|---|
| correct (of 500) | 500 | 500 | 499 | 450 |
| median logit margin | 21.4 | 20.0 | 18.9 | 10.4 |

95% Wilson intervals: 500/500 is [99.2, 100]; 499/500 is [98.9, 100]; 496/500 (99.2%) is [98.0, 99.7]; 478/500 (95.6%) is [93.4, 97.1]; 450/500 (90.0%) is [87.1, 92.3].

![Elementary model, planted chases. Left: accuracy against width at 24 and 40 hops; the shaded region is wider than any training chase. Right: accuracy against hops at width 8, with the share of real training slots having at least that many hops under each tick. Teacher-forced, 500 words per point, 95% Wilson intervals.](fig_elem_cells.png)

**Fresh natural words** (3,000 words, 48,000 slots; `elem_natural.npz`). Slot accuracy is 47,992/48,000 (99.983%) and word accuracy 2,992/3,000 (99.73%). By width:

| width | slots | errors |
|---|---|---|
| 1–7 | 41,231 | 0 |
| 8–10 | 6,179 | 1 |
| 11 | 375 | 4 |
| 12 or more | 215 | 3 |

By hop count the 8 errors are spread across the bins with 10–24 hops (3, 3 and 2 errors in bins of 21,300, 17,537 and 3,596 slots), and there are none among the 213 slots with 25 or more hops. I see no pattern with hop count.

**What it shows.**
- **For this model, width is what matters, not hop count.** At width 8 accuracy is 99.8–100% from 12 to 80 hops. Training chases reach 78 hops, so $L=80$ is only just past the longest one seen and this is not an extrapolation test.
- **Leaving the trained width costs a few points, and more as chains get longer.** At width 16 (five beyond the training maximum of 11) accuracy is 95.6% at 24 hops and 90.0% at 40 hops. The margin falls smoothly (23 to 12 at $L=24$), so the failure is soft, as for the general model (§3.3).
- **Degradation starts at the rare tail of the support, not exactly at its edge.** Width 11 is within the support but holds about 0.4% of training slots, and it has 4 errors in 375 slots (1.1%), against 1 error in 47,410 slots at widths below 11.
- **Contrast with the general model.** There, accuracy fell steeply with hop count at fixed $s$ (§3.3) and depended only weakly on $s$. Here it is nearly flat in hop count and depends on width. The two training sets differ in the way that matters: the general model's training data had essentially no chase of 19 or more hops, whereas the elementary model's had many (3.9% of slots with 40 or more).
- **Caveats on that comparison.** Hops of $\pm1$ inside a window of at most 11 positions are not the same computation as arbitrary jumps among 25 points, and many of the hops in a narrow window are back and forth. The width limit of the elementary model and the hop limit of the general model are both coverage effects of the training data, but I cannot show from this that the general model would behave like the elementary one if it were trained on long chains.

**Position 45 (the block in the position-embedding similarity, Figure 5b).** I looked for structure at position 45 in the real training data (`elem_data_stats.py`, `elem_data_stats.txt`, 400,000 training rows and the test file) and in this checkpoint (`elem_pos_gram.py`).
- *Data.* The training rows are position-stationary: for every position the token histogram is consistent with the pooled one (chi-square $z$ scores between $-1.9$ and $+1.6$ over 120 positions, none beyond 3; $z$ at positions 40–50 between $-0.9$ and $+1.5$), and the mean Coxeter length of the product of the first $i$ letters is smooth through $i=45$ (second differences below 0.01 there). I found nothing special at 45. Training outputs move at most 13 points and have Coxeter length at most 42, while the test file reaches 16 points and length 48 (0.35% of test outputs have length 45 or more, against none in training). The test file's token histogram is flat (0.062 each) while training's is bell-shaped (0.02 at the ends, 0.085 in the middle).
- *Checkpoint.* Mean cosine similarity of position embeddings at lag 1, 5, 20 and 45 is 0.39, 0.13, $-0.01$ and $-0.03$. A "within-block minus cross-block" score for a boundary at 45 is 0.034, lower than boundaries near 15–18 and 25 (0.048–0.055). Weak structure near 45 and 72 is visible in `elem_pos_gram.png`, but I would not call this checkpoint a clear example of the block. So I have not run the position-embedding swap (Test 6b): it would be informative only on a seed that shows the block, and I do not have one.

## 4. What this means for the density idea

- The algebraic reason the target is local (Moore's presentation: every relation involves at most 4 points, and the pointer chase needs one point at a time) stands. The experiments above don't test it directly.
- What the data does say is that the model's generalization beyond the training support is real at the level of whole words (all 2,000 real test words touch 17 or more points, against at most 10 in training), while each output slot only needs a short chase (at most 9 distinct points in the real test file). The ability to follow a chain falls off at $L\approx15$ to 20, which is where the training data runs out of long chains. Crossing the support boundary ($s=10\to11$) costs only 2 to 4 points. An accurate statement of what training "covers" is therefore about **chain length (hop count)** much more than about how many distinct points a word or a trajectory touches.
- In the spirit of the $\mathbb{Q}\subset\mathbb{R}$ analogy: the objects that need to be dense are trajectories (chases), not words. The training distribution is dense in short chains, wherever they sit among 25 points, and sparse in long ones. A density argument would need a metric in which "long chain" is far from "short chain".
- I can't say from this evidence whether the limit is architectural (depth) or statistical (rarity in training). Separating them needs experiments with new training runs (§6).
- **[NEW]** Three results from §3.6–3.8 bear on this. (i) The elementary model, trained on far longer chases than the general model, shows no hop-count cliff up to 80 hops and loses accuracy only beyond the widths it was trained on. That supports reading the general model's cliff as a coverage effect in hop count, though the two tasks differ. (ii) The interventions say the general model is local in the sense the algebra suggests: for $L\le12$ the output depends on the chase letters and nothing else. (iii) The probes say it does not hold the chain's later states at the query position, so how chains of 8–12 hops are composed within five blocks is still open, and a density argument built on chains would need to account for it.

## 5. Corrections to my earlier messages
1. **Ladder hypothesis.** I framed the ladder test around a boundary at 11 distinct points, the first beyond the $m=10$ support. That boundary is correct (§2.3). There is no collapse at it (summary item 5).
2. **Word length.** My earlier closed-form tail estimates assumed 24 letters. The config uses 25 (`MAX_TRANS_NUMBER = 25`). Recomputed, the per-trajectory probability of at least 11 distinct points is $1.1\times10^{-6}$ (was $7.0\times10^{-7}$), and per word at most $2.7\times10^{-5}$ (was $1.7\times10^{-5}$). The simulation I ran earlier used 24 letters; the stratified run here uses 25 and saw none in 20,000 words, consistent with this.
3. **Training support.** I assumed $m=10$, then switched to $m=16$ on the config's say-so, and finally measured the real data: $m=10$ (§2.3). The config's `ACTUAL_GROUP_SIZE` does not describe the data.
4. **Gram spectrum.** See §3.5.
5. **Untested:** my earlier suggestion that $m=4$ or $5$ might already suffice has not been tested (it needs retraining).
6. **[NEW] Leftover in §7.** The first version of this report still described the fixed-hop sweep as crossing an $s=16/17$ boundary in the file table in §7. That was left over from the wrong $m=16$; the sweep crosses $s=10/11$, as §3.3 says. Fixed.

## 6. Not done, and suggested next steps [UPDATED]

**Inference-only tests run in this version:** layer-wise probes (§3.6), irrelevance interventions (§3.7), and the elementary-transposition port with baseline, width and hop-count tests (§3.8, Test 6a).

**Inference-only tests not yet run:** probes (and activation patching) at the input-word positions, which follow from §3.6 and are the way to find where long chains are composed; gap/alignment variations beyond spread versus contiguous; attention patterns; weight-level equivariance; the position-embedding swap for the elementary model (Test 6b), which needs a checkpoint that shows the block.

**Needs training (not possible in my environment):** the support-size sweep; models of different depth; training with long-chain-enriched data.

Given §3.6–3.8, I would prioritise:
1. **Training the general model with long-chain-enriched data** (or a deeper model). This is now more motivated: the elementary model suggests long chains are learnable when present. If the general model's cliff moves with the data, it was coverage.
2. **Probes and activation patching at the input positions** of the general model, to find where chains of 8–12 hops are composed, since the query position does not hold them (§3.6).
3. **Attention patterns of the elementary model at long $L$** (40–80 hops, width 8), to see how a five-block model gets there.

## 7. Reproducing [UPDATED]

Run from this folder (`claude-analysis/`). Scripts find the checkpoint at `../scaling-generator/trained-models/torn-bigger-13/model.safetensors`; set `TORN_MODEL` to override. Needs only `numpy` (plus `matplotlib` for figures); no PyTorch. To get the real data for `data_stats.py`, run `git cat-file blob 5863de5:scaling-generator/data/torn_bigger/<file>` in a clone of `Midataur/permutations` for `train_data1.csv`, `train_data1_perms.csv`, `test_data.csv` and `test_data_perms.csv`.

| file | purpose |
|---|---|
| `st_loader.py`, `np_transformer.py` | safetensors reader and NumPy forward/decoder |
| `permdata.py`, `evalutil.py` | word generation, pointer-chase statistics, planted words, teacher-forced evaluation |
| `sanity.py` | checks the decoder and pointer picture |
| `run_stratified.py N_FREE N_TF` then `analyse_strat.py` | baseline and stratified accuracy (I used `3000 20000`; about 7 minutes) |
| `run_ladder.py N_FREE N_TF` | ladder tests (I used `200 400`; about 5 minutes) |
| `run_beyond_support.py N_TF` | fixed-hop sweep across the $s=10$/$11$ boundary (I used `1000`; about 3 minutes) **[UPDATED]** (the first version of this table said $s=16/17$, left over from the wrong $m=16$) |
| `support_stats.py` | how many points fresh test words touch and move (seconds) |
| `data_stats.py DATA_DIR` | statistics of the real training file and test file, and the model on the real test file (§2.3; needs the data, see below) |
| `run_error_anatomy.py` | failure anatomy (§3.4) |
| `run_gram.py` | Gram spectrum (§3.5) |
| `make_figure.py` | draws `fig_accuracy_vs_hops.png` from `ladder_results.csv` |
| **[NEW]** `probe_layers.py N_WORDS`, `make_probe_figure.py` | layer-wise probes (§3.6; I used `4000`; about 6 minutes), `fig_probe_layers.png` |
| **[NEW]** `run_irrelevance.py N_WORDS N_MAP`, `make_influence_figure.py` | irrelevance interventions (§3.7; I used `800 300`; about 12 minutes), `fig_influence_map.png` |
| **[NEW]** `elem_tools.py` | helpers for the elementary model: trajectories, widths, planted chases, evaluation |
| **[NEW]** `elem_stats.py DATA_DIR` | real elementary training and test statistics, and the model on the real test file (§3.8; needs the data) |
| **[NEW]** `run_elem_cells.py N_PER_CELL N_NATURAL` | planted cells and natural-word accuracy by width (§3.8; I used `500 3000`; about 8 minutes) |
| **[NEW]** `make_elem_figure.py` | draws `fig_elem_cells.png` |
| **[NEW]** `elem_data_stats.py DATA_DIR`, `elem_pos_gram.py` | position-45 checks on the elementary data and checkpoint (§3.8) |

**[NEW]** The elementary data is in `Midataur/permutations` at commit `fdefb8c`, folder `scaling-generator/data/elem_partitioned_long_boosted/`; fetch `train_data1.csv`, `train_data1_perms.csv`, `test_data.csv` and `test_data_perms.csv` with `git cat-file blob` as above. The elementary checkpoint is `elem-partitioned-long-boosted-1/model.safetensors` in the same `trained-models` folder as the general one.

Results included: `ladder_results.csv`, `beyond_support_results.csv`, `support_stats.txt`, `data_stats.txt`, `strat_summary.txt`, `gram_results.json`, `gram_spectrum.png`, `fig_accuracy_vs_hops.png`. **[NEW]:** `probe_results.csv`, `probe_controls.json`, `probe_log.txt`, `irrelevance_results.json`, `influence_map.json`, `irrelevance_log.txt`, `elem_stats.txt`, `elem_cells_results.csv`, `elem_cells_log.txt`, `elem_data_stats.txt`, `elem_pos_gram.png`, `fig_probe_layers.png`, `fig_influence_map.png`, `fig_elem_cells.png`, and the arrays `elem_train_chase_hist.npz` and `elem_natural.npz`. The raw per-slot arrays from the stratified run (`strat_results.npz`, 2.3 MB) are not included; `run_stratified.py` regenerates them with the same seed.
