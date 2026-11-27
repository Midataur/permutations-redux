# Faithful replication of `torn-bigger-13`, with a snapshot after every epoch

One 4-GPU job that reruns the original training as closely as I can make it, to find out whether the cliff at 15 to 20
hops reproduces and *when* it forms. The previous experiment (`../hpc-experiment/`) changed too many things at once
(fresh data, batch, schedule, seed, length); this one changes nothing that affects training.

```
git pull                                 # in your permutations-redux clone
bash hpc-replication/setup_env.sh        # once: Python 3.9 venv + accelerate config matching the original run
bash hpc-replication/submit.sh           # on the Spartan login node
```

`setup_env.sh` builds `replication-runs/env-torn13/` without touching your existing venv or your accelerate default
config: a Python 3.9 venv with torch 1.13.1, accelerate 0.32.1, deepspeed 0.14.4 and every other package pinned to the
original run's versions (`original_requirements.txt`, from wandb), and `accelerate_config.yaml` with 4 processes,
DeepSpeed and bf16. `submit.sh` then uses both automatically. The original run's DeepSpeed settings were not logged
(`"deepspeed": "enabled"` in wandb is a fixed string in `training.py`), so the config uses ZeRO stage 2 without gradient
clipping, the defaults `accelerate config` offers; `ZERO_STAGE=... GRAD_CLIP=... bash setup_env.sh` changes them.

When the job ends, upload `replication-runs/<run>/replication-<run>.zip`. Nothing is pushed to git.

## What is the same as the original run

Everything that wandb and the repo record for `torn-bigger-13`:

* **code, config and data of commit `5863de5`** ("torn-bigger-13", 12 Feb 2025, 40 minutes before the run started).
  If there is no clone of `Midataur/permutations` at `$REPO`, `submit.sh` makes one itself in
  `replication-runs/permutations-history` (history without file contents; fetching the 46 data files makes it about
  3.3 GB, because git also fetches the older data versions they are stored as deltas against; it can be deleted after
  submitting, or kept for further runs). The run itself logged commit `d85cbd4`, which is not on GitHub; `5863de5` is the last pushed commit before it and
  its config is the released `torn-bigger-13-config.py`. The data files are taken out of your repo's history with
  `git show` (git checks their hashes); the job also records their sha256.
* `accelerate launch main.py` with your accelerate default config, as in `transformer-fos.slurm`: 4 L40S GPUs,
  DeepSpeed, bf16, 1,024 words per GPU (4,096 per step), `ReduceLROnPlateau` on the training loss, seed 42.
* the package versions of the original run, in a separate venv made by `setup_env.sh` (your existing virtualenv,
  `$REPO/scaling-generator/venv`, is used only if that has not been run).

## What is different, and why

| change | why |
|---|---|
| `MODELNAME = "torn-replica-s42-<date>"` | `training.py` resumes from `model/<MODELNAME>/model.safetensors` if it exists, and wandb resumes a run with the same id: a new name guarantees a fresh start |
| `num_epochs = 120` (original: unlimited; the run went to 161, then 24 more after an accidental resume) | to fit the wall time; memorisation (epochs 81-100) and most learning-rate drops happen before 120 |
| model name `torn-replica-s42-<date>` | also the wandb run id, so every submission is a new run; wandb logs online to your `scaling-generator` project as the original did, so progress can be watched live (`WANDB_MODE=offline` if the compute node has no network; the local run folder goes into the zip either way) |
| a watcher process copies the model file after every epoch | `training.py` overwrites it each epoch; the copies are the point of the run. The training code is not touched |

## Checks before submitting

`submit.sh` stops (unless `CONFIRM=1`) if:
* torch, accelerate or deepspeed in the venv differ from the original run's versions (torch 1.13.1, accelerate 0.32.1,
  deepspeed 0.14.4, from the wandb `requirements.txt`, kept here as `original_requirements.txt`);
* the accelerate default config is not 4 processes, DeepSpeed and bf16, or uses gradient accumulation.

It prints a table of the differences, so you can decide.

## Resources

4 GPUs (L40S, `fos-gpu-l40s`, as the original), 4 CPUs, 160 GB memory, 8 hours.

* **Time:** the original took 165-167 s per epoch on 4 L40S, so 120 epochs is about 5.6 hours plus data loading.
  Its resumed epochs took 288 s; at that speed 120 epochs would not fit, so training is stopped 25 minutes before the
  wall time and the zip is still written (snapshots up to then). `WALL=07:00:00` or `EPOCHS=100` make it shorter.
* **Memory:** `MaskedDataset` builds Python lists of all 8M words in each of the 4 processes: I measured about 1.6 GB
  per 400,000 words at the peak, so about 30 GB per process and 120 GB for four. 160 GB leaves a margin (the original
  script asked for 700 GB).
* **Disk:** about 1.5 GB of data and 2.5-5 GB of snapshots in the run folder.

## Output

`replication-<run>.zip`: the training log (progress bars removed), Slurm output, `env/` (config used, `pip freeze`,
accelerate config, GPUs, data checksums), the offline wandb run, and snapshots after epochs 1-12, every 4th epoch
after that, and the last one (all snapshots stay in the run folder; `python make_zip.py RUN NAME 1` packs all of them).

## What I will do with it

At every snapshot: the planted hop-count curve ($L$ = 8 to 24), and Test 16 (training slots against fresh slots of the
same hop count). With the training and validation loss from wandb, this shows whether the cliff reproduces, and
whether it forms when the training loss collapses (the original: epochs 81-100).

## Tested

On CPU, with the code and config of `5863de5`, 4,000 training words, batch 64 and 3 epochs: `submit.sh` up to the
environment check (with a fake `sbatch`), then the job script end to end (training, watcher, zip). The snapshots
load in `claude-analysis/np_transformer.py`. Not tested: the GPU run, DeepSpeed, your accelerate config and Spartan's
`module`/`sbatch`.
