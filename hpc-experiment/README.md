# One job: coverage vs depth, and embedding pictures during training

Everything runs from **one** `sbatch`. It trains four models in parallel (one per GPU), takes snapshots of the position and
token embeddings while they train, scores each snapshot on fixed test sets split by number of hops, and writes **one zip**
for you to upload. Nothing is pushed from the cluster.

```
cd hpc-experiment        # after a git pull
sbatch run.slurm
```

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

`run.slurm` loads the same modules (`PY_MODULES`, empty to skip), then uses `./venv` (or `$VENV`, or `../scaling-generator/venv`),
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

| arm | GPU | what it is | what it tells you |
|---|---|---|---|
| `g_m10` | 0 | the paper's distribution: natural words over 10 points | control. Should reproduce the cliff at 15 to 20 hops |
| `g_plant10` | 1 | 50% natural + 50% words with one planted chain of 3 to 25 hops through 2 to 10 points; the whole word stays on 10 points | **coverage**: does the cliff move when long chains are in the data? |
| `g_plant10_b8` | 2 | same data, 8 blocks | **depth**: does more depth remove the cliff? |
| `e_base` | 3 | the elementary-transposition model (Figure 5b): 120 letters, S_16, partitioned windows | **embedding pictures during training**: when do the position-embedding blocks (72 and 120 in Figure 5b) form? |

Optional extras (`ARMS="..."`): `g_plant10_b3` (3 blocks), `g_plant5` (planted chains on 5 points only: how much support
is needed?), `g_m5` (natural words on 5 points). More arms than GPUs share GPUs, so each gets fewer steps in the same time.

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

`hpc-outputs/hpc-run-<jobid>.zip` (about 15 MB) is what I need. It has `summary.json` (final numbers), per arm
`metrics.jsonl`, `pos_emb.npz`, `config.json`, `train.log`, the first pictures, `env.json`, `plan.json`, `orchestrator.log`.
`hpc-run-<jobid>-extras.zip` (about 200 MB: checkpoints, token embeddings, the evaluation sets) is optional; I can run all my
inference-only tests on those checkpoints (they are in the repo's format), so keep it.

The last lines of the Slurm output file print the path of the zip.

## Time

I could not measure GPU speed from here. In its first two minutes the job times 60 seconds of real training for every arm (all
at once, so contention is included), then sets the number of steps so that training takes `TRAIN_HOURS` (default 8). All
general arms get the same number of steps. Total run time is about `TRAIN_HOURS` + 30 minutes. `#SBATCH --time=0-10:00:00` and
`WALL_HOURS=10` go together: if you change one, change the other. For a first look, `TRAIN_HOURS=3 sbatch run.slurm`.
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

So an arm needs roughly 160M words to be converged, and the plateau lasts 70M words or fewer. After calibration the job prints, for every arm, steps x batch = words and warns if that is below 1.5 times 160M. If your GPUs run as fast as in your logs, 8 hours gives about 345M words to a general arm (less for the 8-block arm) but only about 115M to the elementary arm, which is a little short of converged; it will still show the early phase of the position-embedding pictures. To give every arm more: `TRAIN_HOURS=14 WALL_HOURS=16 sbatch --time=0-16:00:00 run.slurm`. The batch is 1,024 words per step (your runs: 4,096); `BATCH=2048 sbatch run.slurm` changes it, if the memory of the elementary arm allows.

**Safety.**
* A failing arm fails in the calibration minute, before any real training, and the zip then contains the logs.
* 25 minutes before the time limit Slurm sends a warning, and the arms stop cleanly, take a last snapshot, and the zip is still written.
* Resume: `RESUME_DIR=hpc-outputs/run-<jobid> sbatch run.slurm` continues every arm from its last `resume.pt` (saved every 20 minutes and at a clean stop).

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
and a tiny end-to-end run of the whole pipeline.

## Files

`run.slurm` (the one command), `run_all.py` (orchestrator), `exp/` (`model.py`, `datagen.py`, `evalsets.py`, `evaluate.py`,
`train.py`, `arms.py`, `collect.py`, `figures.py`, `np_tools.py`), `tests/`, `export_wandb.py` (dumps wandb run histories to a zip).
