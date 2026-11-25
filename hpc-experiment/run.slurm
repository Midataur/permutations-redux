#!/bin/bash
# Coverage-vs-depth experiment + embedding pictures during training.  ONE job, FOUR arms (one per GPU).
# Submit with:   cd hpc-experiment && sbatch run.slurm
# Output:        hpc-outputs/hpc-run-<jobid>.zip   (upload this)      [+ ...-extras.zip, optional]
# See README.md for what it does, how to shorten it, and how to resume it.

# Partition for the job (same as scaling-generator/transformer-fos.slurm):
#SBATCH --partition=fos-gpu-l40s
#SBATCH --qos=fos

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="perm-coverage-vs-depth"

# The project ID which this job should run under:
#SBATCH --account="punim2163"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

# Number of GPUs requested per node (the default experiment has 4 arms, one per GPU):
#SBATCH --gres=gpu:4
# The amount of memory in megabytes per node (data are generated on the GPU, so this is plenty):
#SBATCH --mem=96000

# Use this email address:
#SBATCH --mail-user=mpetschack@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec.  Keep WALL_HOURS below in step with this.
#SBATCH --time=0-10:00:00

# Slurm sends USR1 to this script 25 minutes before the limit; we then stop the arms cleanly and still write the zip.
#SBATCH --signal=B:USR1@1500

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# ------------------------------------------------------------------ settings you may want to change
# Training time per arm, in hours.  The number of steps is calibrated at the start to fit this.
export TRAIN_HOURS=${TRAIN_HOURS:-8}
# Must match --time above (in hours).
export WALL_HOURS=${WALL_HOURS:-10}
# Which arms to run (one per GPU).  Default: g_m10 g_plant10 g_plant10_b8 e_base.  Extras: g_plant10_b3 g_plant5 g_m5
# More arms than GPUs share the GPUs (each then gets fewer steps in the same time), e.g. the 6-arm version:
# export ARMS="g_m10 g_plant10 g_plant10_b8 e_base g_plant10_b3 g_plant5"
# ------------------------------------------------------------------

# The modules to load:
module load CUDA/12.4.1

# Run the job from the directory where it was launched (default): this folder, hpc-experiment/
cd "$SLURM_SUBMIT_DIR" || exit 1

# Find the python environment you already use for scaling-generator (only torch, numpy and safetensors are needed;
# matplotlib is optional, for the pictures).  Set VENV=/path/to/venv to override.
for cand in "$VENV" ./venv ../scaling-generator/venv ../venv; do
    if [ -n "$cand" ] && [ -f "$cand/bin/activate" ]; then source "$cand/bin/activate"; echo "using venv $cand"; break; fi
done
python - <<'PY' || { echo "ERROR: torch/numpy/safetensors not importable in this environment (set VENV=...)"; exit 1; }
import torch, numpy, safetensors
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, "no GPU visible"
PY
nvidia-smi -L

# The job command(s).  Run in the background so that this shell can forward the USR1 warning to it.
python run_all.py &
PID=$!
trap 'echo "USR1 (time limit approaching): forwarding to run_all.py"; kill -USR1 $PID' USR1
wait $PID
STATUS=$?
# the trap interrupts the first wait; wait again for the clean shutdown
while kill -0 $PID 2>/dev/null; do wait $PID; STATUS=$?; done

# put a copy of this job's stdout next to the zip so you can upload it if something looks wrong
ls -la hpc-outputs/*.zip 2>/dev/null

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
