#!/bin/bash
# One 4-GPU job: the original torn-bigger-13 training code and data (commit 5863de5), unchanged apart from the
# model name and the number of epochs, plus a watcher that keeps a copy of the weights after every epoch.
# Submitted by submit.sh, which sets --time, --chdir and the variables RUN, VENV, MODELNAME, TRAIN_SECONDS, RUN_ID.
#SBATCH --partition=fos-gpu-l40s
#SBATCH --qos=fos
#SBATCH --account=punim2163
#SBATCH --job-name=torn-replica
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=160G

echo "job $SLURM_JOB_ID on $(hostname), $(date)"
module load CUDA/12.4.1
source "$VENV/bin/activate"
set -u
export WANDB_MODE="${WANDB_MODE:-online}"   # online as in the original run, so progress can be watched live;
                                            # WANDB_MODE=offline bash submit.sh if the compute node has no network

nvidia-smi > "$RUN/env/nvidia_smi.txt" 2>&1
python "$RUN/snapshot_sidecar.py" "model/$MODELNAME/model.safetensors" "$RUN/snapshots" 10 &
SIDE=$!

# the original command was `accelerate launch main.py` (transformer-fos.slurm); training stops cleanly
# 25 minutes before the wall time if it has not finished its epochs by then
timeout --signal=INT --kill-after=300 "$TRAIN_SECONDS" accelerate launch ${ACCEL_CONFIG:+--config_file "$ACCEL_CONFIG"} main.py > "$RUN/train_raw.log" 2>&1
echo "training exit code $? at $(date)" | tee "$RUN/train_exit.txt"

touch "$RUN/snapshots/STOP"; wait $SIDE
cp -r wandb "$RUN/wandb" 2>/dev/null
python "$RUN/make_zip.py" "$RUN" "replication-$RUN_ID.zip"
echo "done $(date)"
