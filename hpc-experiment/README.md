# Coverage vs depth, and embedding pictures during training: one command, one single-GPU job per arm

One command submits one **single-GPU job per arm** (four by default). The arms train independently, take snapshots of the
position and token embeddings while they train, score each snapshot on fixed test sets split by number of hops, and the results
end up in **one zip** for you to upload. Nothing is pushed from the cluster.

```
cd hpc-experiment        # after a git pull
bash submit.sh
```

**Why one job per arm.** A job that asks for 4 GPUs has to wait for a whole free node; four jobs that ask for one GPU each can
start on whatever is free, one by one. **The jobs do not depend on each other, so they can start and finish in any order**:

* each job builds the evaluation sets it needs from fixed seeds (the same sets for every arm of a kind) and trains for the same
  fixed number of steps as the other arms of its kind, so nothing has to be measured across arms (`STEPS_G` for the general
  arms, `STEPS_E` for the elementary one, set once when you submit);
* each job writes only into its own folder `hpc-outputs/run-<RUN_ID>/<arm>/`;
* when a job finishes it refreshes the combined zip `hpc-outputs/hpc-run-<RUN_ID>.zip` with whatever exists at that moment
  (arms that are still running appear with their state and the snapshots so far). The zip is replaced atomically and the
  refreshes take turns, so the last job to finish leaves the complete zip, whatever the order was. You can also rebuild it by
  hand at any moment: `bash collect.sh hpc-outputs/run-<RUN_ID>`, or look at the state of each arm with `--status`.

`submit.sh` prints the run folder and the zip name. Settings are environment variables, for example
`STEPS_G=250000 WALL_HOURS=12 bash submit.sh` (all listed at the top of `submit.sh`). The older single 4-GPU job
(`sbatch run.slurm`, `run_all.py`) still works if you have a whole node free.

### Setting up the Python environment (once)

The job needs `torch`, `numpy` and `safetensors`; `matplotlib` is optional (pictures). No wandb, no accelerate, no data
files. With the modules from your newer Slurm script (Python 3.14.2), on a login node:

```
module load CUDA/12.4.1 GCCcore/13.3.0 XZ/5.4.5 Python/3.14.2
cd hpc-experiment
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch numpy safetensors matplotlib
```

`job_arm.slurm` (through `env.sh`) loads the same modules (`PY_MODULES`, empty to skip), then uses `./venv` (or `$VENV`, or `../scaling-generator/venv`),
checks that torch sees a GPU, and stops with a clear message if not. I tested on Python 3.13 with torch 2.15, numpy 2.5 and
safetensors 0.8 (CPU only). Those packages all have wheels for Python 3.14, but I have not run on 3.14 or on a GPU myself. The code
uses only features that exist in torch 2.1 as well, in case you reuse an older environment.

## Why these arms

The general model (`torn-bigger-13`) is accurate up to about 12 hops and falls off between 15 and 20 (report §3.3). Its
training data has essentially no chain that long. Two explanations remain:

* **coverage**: long chains were too rare in training;
* **depth**: five blocks cannot compose that many hops. The new probes at the word positions (report §3.9) show chain
  continuations becoming readable one layer at a time with a reach that roughly doubles per layer (about 1, 1, 4, 8, 13 or more
  hops at layers 1 to 5), which is what log-depth "pointer jumping" would look like, and would cap five blocks near 16 hops.

| arm | job | what it is | what it tells you |
|---|---|---|---|
| `g_m10` | 1 GPU | the paper's distribution: natural words over 10 points | control. Should reproduce the cliff at 15 to 20 hops |
| `g_plant10` | 1 GPU | 50% natural + 50% words with one planted chain of 3 to 25 hops through 2 to 10 points; the whole word stays on 10 points | **coverage**: does the cliff move when long chains are in the data? |
| `g_plant10_b8` | 1 GPU | same data, 8 blocks | **depth**: does more depth remove the cliff? |
| `e_base` | 1 GPU | the elementary-transposition model (Figure 5b): 120 letters, S_16, partitioned windows | **embedding pictures during training**: when do the position-embedding blocks (72 and 120 in Figure 5b) form? |

Optional extras (`ARMS="..."`): `g_plant10_b3` (3 blocks), `g_plant5` (planted chains on 5 points only: how much support
is needed?), `g_m5` (natural words on 5 points). Each extra arm is one more job (`ARMS="g_m10 g_plant10 g_plant10_b8 e_base g_plant5" bash submit.sh`).

### What I expect (written before the run, so that it can be wrong)

* `g_m10`: cliff at 15 to 20 hops, as in `torn-bigger-13`.
* If the cliff is **coverage**: `g_plant10` has accuracy of 90% or more at 20 to 24 hops at every `s`, including `s=14` and `s=20`,
  which is outside the 10-point support.
* If the cliff is **depth**: `g_plant10` still falls off somewhere between 16 and 24 hops, and `g_plant10_b8` does not.
* Mixed outcomes are likely: for example `g_plant10` moves the cliff partly, and `g_plant10_b8` finishes the job.
* `e_base`: no prediction for the timing of the blocks. I would like to see whether they exist at all in a fresh seed.

## What is measured

For every arm, at about 60 snapshots (log-spaced in step, plus evenly spaced ones):

* **accuracy and loss** on fixed evaluation sets (teacher forced), split by hop count and by number of distinct points
  (general) or width (elementary):
  `nat25` (natural words on all 25 points, like the paper's test file), `nat10` (the original training distribution),
  `ladder_full` (planted chains with `L` = 4..24 hops and `s` = 3..20 points, letters over all 25 points),
  `ladder_in10` and `ladder_in5` (the same chains but the whole word on 10 or 5 points); elementary: `nat16`, `train_like`,
  `cells` (width 5..16 x 20..80 hops). Planted sets are scored at the planted slot only.
* the **position embeddings** (every snapshot) and **token embeddings** (about 9 snapshots), so you can draw the similarity
  pictures of Figure 4b and 5b at any time in training, and make gifs;
* training loss and accuracy; the repo-format checkpoint at 25% and 100% of training.

## Output

`hpc-outputs/hpc-run-<RUN_ID>.zip` (about 15 MB) is what I need. It has `summary.json` (final numbers and the state of every arm),
per arm `metrics.jsonl`, `pos_emb.npz`, `config.json`, `train.log`, `job.log`, `env.json`, and the first pictures.
`hpc-run-<RUN_ID>-extras.zip` (about 200 MB: checkpoints, token embeddings, the evaluation sets) is optional; I can run all my
inference-only tests on those checkpoints (they are in the repo's format), so keep it.

The Slurm output of each job is in `hpc-outputs/logs/perm-<arm>-<jobid>.out`; its last lines print the path of the zip.
`summary.json` (and `bash collect.sh ... --status`) say, for each arm, whether it is `done`, `truncated` (stopped at the deadline)
or `failed`; upload the zip when all four are done, or earlier if one job is slow to start and you want a first look.

## After the run: analysing the checkpoints (done off the cluster)

Each arm saves two checkpoints (at 25% and at the end, or where it was stopped at the deadline). They are in the **`-extras` zip**
(`hpc-run-<id>-extras.zip`, a few hundred MB), not in the main one, so upload both zips. Then, from `claude-analysis/` (NumPy only, no GPU):

    python analyze_run.py hpc-run-<id>.zip hpc-run-<id>-extras.zip --out analysis-<id> --patching

It takes the last checkpoint of each arm and runs the planted-chain accuracy sweep (also beyond the 10-point support), the probes at the
word positions (section 3.9), attention patterns and single-head ablation (section 3.12), and with `--patching` activation patching
(section 3.10), about 1.5 hours for all four arms on 2 cores (`--quick` for a rough pass in about 20 minutes). `summary.md` puts the arms
side by side. Finished jobs are skipped, so it can be re-run. The 8-block arm was checked with a checkpoint that is the 5-block model
plus three blocks that add nothing: the results agree with the 5-block ones exactly on the shared blocks.

## Time

I could not measure GPU speed from here, so the number of steps is fixed when you submit: `STEPS_G=200000` for every general arm
(205M words) and `STEPS_E=110000` for the elementary arm (113M words), chosen from the speeds in your logs (below) so that
the slowest arm, the 8-block one, should still fit in the 10-hour limit. The same number for all general arms is what makes them
comparable, and it is what lets the jobs run in any order. Each job starts with `--calib-seconds` (default 30) of real training,
which fails fast if the arm cannot run, and prints the measured speed and how many hours its steps need. If that is more than
the time limit, the job says so, trains until the deadline (20 minutes before the limit), stops cleanly, and you can continue it:
`ARMS=g_plant10_b8 RESUME_DIR=hpc-outputs/run-<RUN_ID> bash submit.sh`. `WALL_HOURS` (default 10) sets the limit of every job.
For a first look, `STEPS_G=60000 STEPS_E=40000 bash submit.sh`.
Batch size 1024 words per step, learning rate 3e-4, AdamW (weight decay 0.05).

### How many steps? (from your two training logs)

Your logs of the original runs (`torn-bigger-13`, `elem-partitioned-long-boosted-1`) give a reference for how much training is enough:

| | `torn-bigger-13` | `elem-partitioned-long-boosted-1` |
|---|---|---|
| words per epoch (steps per epoch) | 8M (1,954 steps of 4,096 words: 1,024 per GPU on 4 GPUs) | 16M (3,907 steps) |
| time per epoch, speed | 165 s, about 12k words/s per GPU | 1,010 s, about 4k words/s per GPU |
| plateau before learning | accuracy 72-73% for epochs 1-9 (about 18k steps, 72M words) | accuracy 63-68% for epochs 2-4 (epoch 1 is not in the log) |
| 90% train accuracy | epoch 11 | epoch 6 |
| 99.9% train accuracy | epoch 20 (about 39k steps, **160M words**) | epoch 10 (about 39k steps, **160M words**) |
| 99.99% train accuracy | epoch 44 | epoch 21 |
| whole run | 161 epochs, about 7.4 hours | 96 epochs, about 27 hours |

So an arm needs roughly 160M words to be converged, and the plateau lasts 70M words or fewer. Each job prints steps x batch = words and notes if that is below 1.5 times 160M (the default for the general arms, 205M, is 1.3 times). If your GPUs run as fast as in your logs, a general arm needs about 4.7 hours for 205M words (more for the 8-block arm) and the elementary arm about 7.8 hours for 113M words, which is a little short of converged; it will still show the early phase of the position-embedding pictures. To give every arm more: `STEPS_G=300000 STEPS_E=150000 WALL_HOURS=16 bash submit.sh`. The batch is 1,024 words per step (your runs: 4,096); `BATCH=2048 bash submit.sh` changes it, if the memory of the elementary arm allows.

**Safety.**
* A failing arm fails in its first minute, before any real training, and the zip then contains its logs.
* 25 minutes before the time limit Slurm sends a warning, and the arm stops cleanly, takes a last snapshot, and the zip is still refreshed.
* One arm can be stopped without touching the others: `touch hpc-outputs/run-<RUN_ID>/<arm>/STOP`.
* Resume: `RESUME_DIR=hpc-outputs/run-<RUN_ID> bash submit.sh` continues the arms in `ARMS` (default: all four) from their last `resume.pt` (saved every 20 minutes and at a clean stop), with the original number of steps. Name the arms to continue only some: `ARMS="g_plant10_b8" RESUME_DIR=... bash submit.sh`.

## How this differs from `scaling-generator/training.py`

* **Data are generated on the GPU, fresh every step**, instead of read from CSV files for many epochs. This removes CPU
  and file-loading limits (6 CPUs for 4 GPUs) and removes memorisation. The generators reproduce your real data (tested, CPU):
  per-slot hop histogram of the natural general words to three decimals; for the elementary data the token frequencies, window
  spans and the (hops, width) histogram (total variation distance 0.010).
* **Learning-rate schedule:** 100 warm-up steps, constant 3e-4 until 60% of the steps, then cosine to 3e-5. You used
  ReduceLROnPlateau on the training loss. A fixed schedule is needed to size the run, and a plateau scheduler can fire during the early plateaus.
* **Fused attention** (`scaled_dot_product_attention`) and bfloat16 autocast. The model is *the same function* as
  `transformer.py`: logits agree to 1e-6 on both configs (`tests/test_equivalence.py`), with the repo's n_embed^-1/2
  attention scale and mask, and it saves checkpoints in the repo's key layout, so `claude-analysis/np_transformer.py` reads them.
* One seed per arm. Arms with the same architecture share the initial weights, so the comparison between them is paired.

## Limits

* The code was tested on CPU only (tiny model, full pipeline, plus the real-size model in forward and backward). GPU
  behaviour is covered by the self-check minute, not by me.
* One seed each. A difference of a few points between arms at one hop count is not evidence.
* `g_plant10` changes the distribution in one more way than "more long chains": half of its words have a planted slot, whose
  other letters avoid the pointer. Comparing `ladder_in10` and `ladder_full` separates the word-level support from the chain.
* The number of steps is a guess about convergence. If the control `g_m10` has not converged (its `nat25` accuracy is below
  99%), the comparison is weaker; the logs show the curve.

## Tests (CPU, `tests/run_tests.sh`, about 6 minutes)

`test_equivalence.py` (model == your `transformer.py` for both configs; saved file readable by the NumPy port),
`test_datagen.py` (generators == NumPy references == real-data statistics; planted chains have exactly the requested `L` and `s`),
`test_independent_jobs.py` (about 7 minutes on CPU: three single-arm jobs started in a scrambled order and overlapping each end up
in one complete zip with no leftovers; an arm stopped through its own `STOP` file is resumed with its original number of steps
and another arm is unaffected), and a tiny end-to-end run of the older 4-GPU pipeline.

## Files

`submit.sh` (the one command), `job_arm.slurm` (one arm, one GPU), `run_arm.py` (what one job does), `collect_run.py` and `collect.sh`
(build the zip from whatever exists, or show the state of each arm), `env.sh` (modules and venv), `run.slurm` and `run_all.py`
(the older single 4-GPU job), `exp/` (`model.py`, `datagen.py`, `evalsets.py`, `evaluate.py`,
`train.py`, `arms.py`, `collect.py`, `figures.py`, `np_tools.py`), `tests/`, `export_wandb.py` (dumps wandb run histories to a zip).
