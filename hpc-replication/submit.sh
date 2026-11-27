#!/bin/bash
# Faithful replication of torn-bigger-13, with a snapshot of the weights after every epoch.
#
# Run on the Spartan login node, from anywhere:
#     bash hpc-replication/submit.sh
# It (1) takes the code, config and data of commit 5863de5 ("torn-bigger-13") out of your permutations repo,
# (2) makes three config edits (model name, number of epochs, optionally the seed), (3) checks the Python
# environment and your accelerate config against the original run, and (4) submits one 4-GPU job.
# The job writes  $OUT_BASE/<run>/replication-<run>.zip  to upload.  Nothing is pushed to git.
#
# Settings (environment variables, all optional):
#   REPO      your clone of Midataur/permutations      (default /data/gpfs/projects/punim2163/permutations)
#   VENV      the virtualenv the original runs used     (default $REPO/scaling-generator/venv)
#   SEED      42 = the original run's seed              (default 42)
#   EPOCHS    epochs to train                           (default 120; the original ran 161 + 24 after a resume)
#   WALL      Slurm time limit                          (default 08:00:00; training is stopped 25 min before it)
#   OUT_BASE  where run folders go                      (default ./replication-runs)
#   CONFIRM=1 submit even if the environment check finds differences from the original run
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/data/gpfs/projects/punim2163/permutations}"
VENV="${VENV:-$REPO/scaling-generator/venv}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-120}"
WALL="${WALL:-08:00:00}"
OUT_BASE="${OUT_BASE:-$PWD/replication-runs}"
COMMIT=5863de5
RUN_ID="$(date +%Y%m%d-%H%M%S)-s${SEED}"
RUN="$OUT_BASE/$RUN_ID"
MODELNAME="torn-replica-s${SEED}-$(date +%m%d-%H%M)"   # also the wandb run id: unique, so no old run is resumed

echo "== replication run $RUN_ID  (seed $SEED, $EPOCHS epochs, wall $WALL)"
# The data files were deleted from the working tree later (commit eea242a) but are still in the history of 5863de5.
# If there is no clone at $REPO, make a small one (history without file contents); git show then downloads only the
# files needed (about 770 MB).
if [ ! -d "$REPO/.git" ] && [ "$(git -C "$REPO" rev-parse --is-inside-work-tree 2>/dev/null)" != true ]; then
  REPO="$OUT_BASE/permutations-history"
  if [ ! -d "$REPO/.git" ]; then
    echo "no clone of Midataur/permutations found; cloning its history (without file contents) into $REPO"
    mkdir -p "$OUT_BASE"
    git clone --filter=blob:none --no-checkout https://github.com/Midataur/permutations.git "$REPO"
  fi
fi
git -C "$REPO" cat-file -e "${COMMIT}^{commit}" 2>/dev/null || { echo "fetching $COMMIT ..."; git -C "$REPO" fetch origin; }
git -C "$REPO" cat-file -e "${COMMIT}^{commit}"

# ---- 1. code and data of commit 5863de5
mkdir -p "$RUN/code" "$RUN/env"
git -C "$REPO" archive "$COMMIT" scaling-generator ':!scaling-generator/data' | tar -x -C "$RUN/code"
SG="$RUN/code/scaling-generator"
D="$SG/data/torn_bigger"; mkdir -p "$D"
FILES="val_data.csv val_data_perms.csv test_data.csv test_data_perms.csv"
for i in $(seq 1 20); do FILES="$FILES train_data$i.csv train_data${i}_perms.csv"; done
for f in $FILES; do git -C "$REPO" show "$COMMIT:scaling-generator/data/torn_bigger/$f" > "$D/$f"; done
n=$(cat "$D"/train_data{1..20}.csv | wc -l)
[ "$n" = 8000000 ] || { echo "expected 8,000,000 training words, found $n"; exit 1; }
(cd "$D" && sha256sum $FILES) > "$RUN/env/data_sha256.txt"
echo "data: 20 training files (8,000,000 words), validation and test files from $COMMIT"

# ---- 2. config edits (everything else is the config of the original run)
python3 "$HERE/edit_config.py" "$SG/config.py" "$MODELNAME" "$EPOCHS" "$SEED"
cp "$SG/config.py" "$RUN/env/config_used.py"

# ---- 3. environment check against the original run
set +e
(
  source "$VENV/bin/activate"
  python -m pip freeze > "$RUN/env/pip_freeze.txt" 2>/dev/null
  accelerate env > "$RUN/env/accelerate_env.txt" 2>&1
)
cfg="${HF_HOME:-$HOME/.cache/huggingface}/accelerate/default_config.yaml"
[ -f "$cfg" ] && cp "$cfg" "$RUN/env/accelerate_default_config.yaml"
python3 "$HERE/check_env.py" "$RUN/env" "$HERE/original_requirements.txt"
ok=$?
set -e
if [ $ok -ne 0 ] && [ "${CONFIRM:-0}" != 1 ]; then
  echo "The environment differs from the original run (see above). Fix it, or rerun with CONFIRM=1 to submit anyway."
  echo "Prepared folder kept at $RUN"
  exit 1
fi

# ---- 4. submit: training stops 25 minutes before the wall time so the zip is always written
IFS=: read -r hh mm ss <<< "$WALL"
TRAIN_SECONDS=$(( 10#$hh * 3600 + 10#$mm * 60 + 10#$ss - 1500 ))
cp "$HERE/snapshot_sidecar.py" "$HERE/make_zip.py" "$RUN/"
jid=$(sbatch --parsable --time="$WALL" --chdir="$SG" \
      --output="$RUN/slurm-%j.out" \
      --export=ALL,WANDB_MODE="${WANDB_MODE:-online}",RUN="$RUN",VENV="$VENV",MODELNAME="$MODELNAME",TRAIN_SECONDS="$TRAIN_SECONDS",RUN_ID="$RUN_ID" \
      "$HERE/job.slurm")
echo "$jid" > "$RUN/job_id.txt"
echo "submitted job $jid; output folder $RUN"
if [ "${WANDB_MODE:-online}" = online ]; then echo "live progress: wandb project scaling-generator, run $MODELNAME"; fi
echo "when it ends, upload $RUN/replication-$RUN_ID.zip"
