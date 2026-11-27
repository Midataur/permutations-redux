# Survey of the Spartan model folder (27 November 2026)

The model folder from Spartan (`model-snapshots.zip`, 206 files, 6.4 GB uncompressed; deleted from the repo again in
the commit that adds this survey, still in git history) holds the final checkpoint of every run since mid-2024. Each
file is the last epoch saved, because `training.py` overwrites it every epoch. I can't see how long each run trained:
that is in wandb. Configs are taken from the git history of `Midataur/permutations` (`configs_from_git.json`; the
commit where the model name first appears, cross-checked by date).

Files here and in `claude-analysis/`:

| file | what |
|---|---|
| `snapshot-survey/inventory.json`, `inventory.py` | architecture of every checkpoint read from its header (blocks, heads, vocabulary, positions, dtype, date) |
| `snapshot-survey/configs_from_git.json`, `configs_from_git.py` | seed, data folder, group sizes, blocks, learning rate, ... of every commit of `Midataur/permutations` |
| `block_survey.py`, `block_survey.json`/`.txt`, `fig_block_survey.png` | position-embedding block structure of every elementary model |
| `make_block_boundary_figure.py`, `fig_block_boundaries.png` | the ten trained models, with the boundary profile |
| `elem_model_survey.py`, `elem_model_survey.json` | how well each elementary model is trained (test file, planted chains) |
| `block_attention.py`, `block_attention.json`, `fig_block_attention.png` | does attention avoid crossing the boundary |
| `block_crossing_test.py`, `block_crossing_results.json` | does a chain across the boundary cost accuracy |
| `cliff_survey.py`, `cliff_survey.json`, `cliff_survey_log.txt` | the hop-count cliff and the memorisation gap across the general models |

Two things had to change in `np_transformer.py` to read the older checkpoints (the newer ones give the same numbers
as before):

* **Legacy architecture.** Every model before `elem-partitioned-long-boosted-1` (11 Feb 2025) has one extra
  multi-head attention layer, `sa_heads`, applied to the embeddings before the blocks, with no layer norm and no
  residual connection (`transformer.py` at commit `825be9c`; `LEGACY_ARCHITECTURE` in later configs). So "4 blocks"
  in those runs is the extra attention layer plus 4 blocks.
* **Masks.** Some older runs were trained with other attention masks, which the checkpoints store. The code now uses
  the stored mask whenever it differs from the current one.

## 1. Block structure in the elementary models

**Question (from your README):** do runs configured like `elem-partitioned-long-boosted-1` also have the block
structure, and was the `rerun` series a test of this?

**What I measured.** For each model with the elementary layout (120 word positions and 16 output positions), the
cosine matrix of the position embeddings, and a boundary detector: for each $k$, how different rows $k$ and $k+1$ of
that matrix are, over all columns more than 6 positions from both rows, as a robust $z$-score against all $k$. A block
boundary shows up as one large $z$ (Figure 5b's model: $z \approx 5$ at 72–73). Untrained seed-42 models have $z\le2.5$
everywhere, with the same small peaks at 25, 80 and 92 (their shared random initialisation).

**How well each run is trained** (`elem_model_survey.py`): teacher-forced accuracy on the first 1,000 words of
`boosted-1`'s real test file, and on 200 planted chains per cell.

| model | seed | architecture | test slots (%) | test words (%) | width 8, 80 hops | width 16, 24 hops | boundary (largest $z$) |
|---|---|---|---|---|---|---|---|
| `elem-partitioned-long-boosted-1` (Figure 5b) | 42 | 5 blocks | 99.99 | 99.9 | 0.99 | 0.98 | **72–73 ($z$ 4.9)** |
| `elem-partitioned-long-boosted-2` | 43 | 5 blocks | 100.00 | 100.0 | 0.98 | 0.95 | **81 ($z$ 5.8)** |
| `elem-partitioned-long-boosted-3` | 44 | 5 blocks | 33.89 | 0.0 | 0.41 | 0.04 | none (still on the plateau) |
| `elem-partitioned-long-boosted-4` | 97 | 5 blocks | 100.00 | 100.0 | 0.99 | 0.98 | none (3.7 at 114, near the end) |
| `rerun-1` (Dec 2025) | 42 | 5 blocks | 99.72 | 95.6 | 0.83 | 0.53 | none (3.1 at 80) |
| `rerun-2` (Dec 2025) | 42 | 5 blocks | 99.91 | 98.6 | 0.97 | 0.90 | weak, 79 ($z$ 3.9, a broad bump over 72–85) |
| `rerun-3` (Dec 2025) | 43 | 5 blocks | 99.99 | 99.9 | 0.97 | 0.93 | **78 ($z$ 7.3)** |
| `rerun-4` (Dec 2025) | 44 | 5 blocks | 99.99 | 99.8 | 0.98 | 0.98 | none |
| `e_base` (my Spartan run) | 0 | 5 blocks | 99.58 | 93.3 | 0.96 | 0.66 | none |
| `elem-partitioned-long-14` | 42 | legacy + 4 | 98.28 | 77.8 | 0.85 | 0.28 | **69 ($z$ 15.6)**, the sharpest of all |
| `elem-partitioned-long-11`, `-12` | 42 | legacy + 4 | 98.2–98.3 | 78–80 | 0.94–0.96 | 0.32–0.35 | none |
| `elem-partitioned-long-5` | 42 | legacy + 4 | 98.52 | 81.4 | 1.00 | 0.73 | none; its position embeddings are essentially unchanged from initialisation (correlation 0.99995 with the untrained `long-6`, same seed) |
| `elem-partitioned-long-6` to `-10`, `-13` | 42 | legacy + 4 (8 for -8, -9) | 34–64 | 0 | 0.34–0.46 | 0.02–0.10 | none (plateau) |

The `long` series used the older data (`elem_partitioned_long`, actual group size 9), weight decay 0.01, and learning
rate $3\times10^{-5}$ up to `-9` and $3\times10^{-4}$ from `-10`. The `boosted` and `rerun` runs used the boosted data
(group size 10), weight decay 0.05, learning rate $3\times10^{-4}$ and 5 standard blocks: the same config as
`boosted-1` apart from the seed. The `rerun`s were trained in December 2025 with the code and environment of that
time (checkpoints saved in float32, where February's were bfloat16; your accelerate default config is now 2 GPUs
without DeepSpeed). So `rerun-1` and `rerun-2` are not exact reruns of `boosted-1` even with the same seed.

**What it shows.**
- **A block boundary is common, but it is not at a fixed place.** Of the seven fully trained runs with
  `boosted-1`'s config, three have a clear boundary (72, 81 and 78), one a weak one (about 79) and three none. With the
  older data and architecture, `long-14` has the sharpest boundary of all, at 69. The places cluster between 69 and 81
  (58–68% of the way through the word) but differ from run to run.
- **The seed does not fix it.** Seed 43 gave a boundary in February (81) and in December (78). Seed 42 gave 72 in
  February and none or a weak one in December. Seed 44 never trained in February and gave none in December. With the
  code and environment changing in between, I can't separate the seed from the setup.
- **Having a boundary does not go with being better or worse trained.** `boosted-4` and `rerun-4` are as accurate as
  `boosted-1` without one, and the width-16 test does not follow the boundary either.
- **So the boundary is a structure that training often, but not always, settles into,** not a property of the data
  (which are position-stationary, report §3.8) or of the config. This fits §3.15: my `e_base`, with 1/14 of the
  original's updates, had none yet.

**The network uses the boundary in its first layers** (`block_attention.py`, 150 test words). For each head, I
compared the attention between word positions at most 6 apart that straddle a place $c$ with the head's average over
all $c$:

- In every model with a boundary, several heads of block 1 (and some of block 2) hardly attend across it. Their ratio
  at the boundary:
  - `boosted-1`: b1h0 0.13, b2h2 0.22, b1h3 0.41, b2h4 0.40 at 72;
  - `boosted-2`: b1h5 0.05, b1h2 0.34 at 80;
  - `rerun-3`: b1h5 0.14 at 78;
  - `long-14`: all six heads of block 1 at 0.04–0.56 at 69.
- The controls are not completely clean. `boosted-4`, which has no boundary, has no block 1 or block 2 dip near 72;
  its one block-1 dip is at 100 (b1h3, 0.35), and its others are in blocks 3 and 4, spread over 38 to 76. `rerun-4`,
  which also has no boundary in its position embeddings (largest $z$ 2.9 at 107), still has one sharp block-1 dip, at
  82 (b1h0, 0.18). So a single head can split the word without the position table showing it, but only the models with
  a boundary have several early heads dipping at the same place.

So the first layers treat the two sides of the word separately.

**But a chain across the boundary is not harder** (`block_crossing_test.py`, 24 hops, width 12, all hop letters inside
40 positions, 300 words per cell). Chains centred on the boundary are as accurate as chains centred in the left block
(`boosted-1`: 1.000 and 1.000; `boosted-2`: 0.973 and 0.997; `rerun-3`: 0.937 and 0.997; `long-14`: 0.630 and 0.697).
The clearest effect is that chains placed late in the word, among the first letters the chase reads, are harder in
some models (0.81–0.82 for `boosted-2` and `rerun-3` at positions 80–120), and that is about position, not the
boundary. The later blocks bridge the split, as the position-swap test of §3.8 also suggested.

## 2. The general models: the hop-count cliff

**`torn-bigger-13`: the repo checkpoint is the February 2025 model, not the resumed one.** The file on Spartan was
saved in float32 on 29 Nov 2025, after the 24-epoch resume. The repo copy is bfloat16, like every February save, and
every trained weight differs from the Spartan file (the 30 identical tensors are the stored attention masks). So
everything in the report is about the February model, which the paper presumably used. My earlier statement that the
analysed checkpoint was "almost certainly epoch 185" was wrong.

**The resume did not change the cliff** (`cliff_survey.py`, 200 planted words per $L$; natural slots: 300 per $h$,
fewer at $h\ge16$):

| | $L=14$ | 16 | 18 | 20 | 24 | fresh slots $h=15$ | $h=16$ | $h=17$ | own training slots $h=14$–17 |
|---|---|---|---|---|---|---|---|---|---|
| repo (Feb 2025) | 0.99 | 0.89 | 0.72 | 0.49 | 0.35 | 0.940 | 0.937 | 0.800 | 1.000 (all) |
| Spartan (after resume) | 0.99 | 0.94 | 0.77 | 0.49 | 0.34 | 0.960 | 0.925 | 0.867 | 1.000 except one slot at $h=15$ |

This confirms Test 16 on the February model with independent words: training slots are right at every hop count,
fresh ones fall from about 13 hops.

**The other `torn-bigger` runs don't help.**
- `torn-bigger-5` to `-8` and `-10` to `-12` are still on the plateau: about 73% of slots right on their own training
  words, 15–30% at the planted slot. They used the legacy architecture, learning rate $3\times10^{-5}$ (one at
  $3\times10^{-6}$) and 14 or 18 data files; `-11` and `-12` have 8 blocks.
- `torn-bigger-9` gets 100% everywhere, including 24 hops, but its stored mask lets the word positions attend to the
  output positions. In teacher forcing those hold the true permutation entries, so the answer leaks. Its 100% is not
  generalisation, and the next run (`-10`) has the corrected mask.
- `torn-bigger-1` to `-4` predate the masked model and are left out.

So there is no second general model trained on this task that both learned it and kept the current mask, and the
February `torn-bigger-13` stays the only one with the cliff.

## 3. Not analysed

The `torn-big-*` (20-letter words), `torn-masked-*` (10-letter words on 16 points), `hybrid-*`, `bin-medium-*`,
`small-*`, `window-*` and `.pth` models are other problem sizes or tokenisations. The `reverse-*` models (permutation
to word) have position embeddings that are still flat, and I did not evaluate them.

## 4. What would make this firmer

- **wandb histories** for `boosted-2`, `-3`, `-4`, `rerun-1` to `-4` and `long-5`, `-11`, `-12`, `-14` (the same
  export script as before, with these run names). They would say how many epochs each run trained and when it left the
  plateau, which is the missing variable for "boundary or not".
- **Snapshots during training** (as in the replication run) would show when a boundary forms and whether it moves.
