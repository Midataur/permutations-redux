#!/bin/bash
# ONE arm of the coverage-vs-depth experiment, on ONE GPU.  Do not submit this by hand: run   bash submit.sh   (it submits one of
# these per arm, with the right settings).  The jobs are independent: they may start and finish in any order, and each one
# refreshes the combined zip when it finishes.  See README.md.

# Partition for the job (same as scaling-generator/transformer-fos.slurm):
#SBATCH --partition=fos-gpu-l40s
#SBATCH --qos=fos

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1

# The name of the job (submit.sh sets "perm-<arm>"):
#SBATCH --job-name="perm-arm"

# The project ID which this job should run under:
#SBATCH --account="punim2163"

# Maximum number of tasks/CPU cores used by the job (data are generated on the GPU, so little CPU is needed):
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# ONE GPU per job (a 4-GPU request has to wait for a whole free node):
#SBATCH --gres=gpu:1
# The amount of memory in megabytes per node:
#SBATCH --mem=24000

# Use this email address:
#SBATCH --mail-user=mpetschack@student.unimelb.edu.au

# Send yourself an email when the job aborts abnormally (fails) or ends successfully (not at the start: there are several jobs)
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

# The maximum running time of the job (submit.sh passes --time from WALL_HOURS, default 10 hours; keep them in step).
#SBATCH --time=0-10:00:00

# Slurm sends USR1 to this script 25 minutes before the limit; training then stops cleanly and the zip is still refreshed.
#SBATCH --signal=B:USR1@1500

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system: use  bash submit.sh"
   exit 1
fi
: "${ARM:?ARM is not set: use  bash submit.sh}" "${RUN_DIR:?RUN_DIR is not set: use  bash submit.sh}" "${STEPS:?STEPS is not set: use  bash submit.sh}"

LOAD_CUDA=1
source "$SLURM_SUBMIT_DIR/env.sh"          # submit.sh runs sbatch from this folder
python - <<'PY' || { echo "ERROR: torch/numpy/safetensors not importable in this environment (set VENV=...)"; exit 1; }
import torch, numpy, safetensors
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, "no GPU visible"
PY
nvidia-smi -L

# Run in the background so that this shell can forward the USR1 warning to it.
python run_arm.py --arm "$ARM" --run-dir "$RUN_DIR" --steps "$STEPS" --wall-hours "${WALL_HOURS:-10}" ${RESUME:+--resume} &
PID=$!
trap 'echo "USR1 (time limit approaching): forwarding to run_arm.py"; kill -USR1 $PID' USR1
wait $PID
STATUS=$?
# the trap interrupts the first wait; wait again for the clean shutdown
while kill -0 $PID 2>/dev/null; do wait $PID; STATUS=$?; done

# Job monitor command to list the resource usage (the cluster's template puts this last; here it comes before the exit so that
# a failed arm is reported as failed instead of taking the exit status of my-job-stats)
command -v my-job-stats >/dev/null && my-job-stats -a -n -s
exit $STATUS
