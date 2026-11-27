#!/bin/bash
# Build a Python environment and an accelerate config that match the original torn-bigger-13 run, without touching
# your existing venv or your accelerate default config.
#
# Run once on the Spartan login node (it needs the internet for pip):
#     bash hpc-replication/setup_env.sh
# Then submit as usual; submit.sh uses this environment and config automatically when they exist:
#     bash hpc-replication/submit.sh
#
# What it makes, in $ENV_DIR (default hpc-replication/replication-runs/env-torn13):
#   venv/                     Python 3.9 (as the original run: /usr/bin/python3 3.9.18) with torch 1.13.1,
#                             accelerate 0.32.1, deepspeed 0.14.4, numpy 1.26.3, ... pinned to the versions in the
#                             original run's wandb requirements.txt (original_requirements.txt)
#   accelerate_config.yaml    4 processes, DeepSpeed, bf16; ZeRO stage $ZERO_STAGE (default 2), no gradient clipping
#
# The original run's DeepSpeed settings were not logged: training.py writes "deepspeed": "enabled" into wandb as a
# fixed string, and the log has no DeepSpeed lines. Stage 2 without clipping is what `accelerate config` offers by
# default. If you remember other settings, set ZERO_STAGE=... and/or GRAD_CLIP=... (e.g. 1.0).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-$HERE/replication-runs/env-torn13}"   # same place whichever folder you run it from
ZERO_STAGE="${ZERO_STAGE:-2}"
GRAD_CLIP="${GRAD_CLIP:-none}"

# ---- Python 3.9 (torch 1.13.1 has no wheels for Python 3.11 or later)
PY=""
for p in python3.9 /usr/bin/python3 python3; do
  if command -v "$p" >/dev/null 2>&1 && [ "$("$p" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" = 3.9 ]; then
    PY="$(command -v "$p")"; break
  fi
done
[ -n "$PY" ] || { echo "no Python 3.9 found (the original run used /usr/bin/python3 3.9.18); try 'module avail Python'"; exit 1; }
echo "Python: $PY ($("$PY" --version 2>&1))"

mkdir -p "$ENV_DIR"
[ -d "$ENV_DIR/venv" ] || "$PY" -m venv "$ENV_DIR/venv"
source "$ENV_DIR/venv/bin/activate"
export PIP_NO_CACHE_DIR=1          # keep the ~3 GB of wheels out of your home quota

# constraints: every package version of the original run (lines that are not plain name==version are dropped)
grep -E '^[A-Za-z0-9_.\-]+==[^ ;@]+$' "$HERE/original_requirements.txt" | grep -v -E '^(pip|setuptools)==' \
  > "$ENV_DIR/constraints.txt"
python -m pip install -q --upgrade "pip<24.1"
python -m pip install -c "$ENV_DIR/constraints.txt" "torch==1.13.1"
python -m pip install -c "$ENV_DIR/constraints.txt" \
  "accelerate==0.32.1" "deepspeed==0.14.4" "numpy==1.26.3" "scipy==1.11.4" "scikit-learn" "tqdm" \
  "wandb==0.16.1" "safetensors==0.4.3" "setuptools<70"
python - <<'EOF'
import torch, accelerate, deepspeed, numpy, wandb
print("installed: torch", torch.__version__, "| accelerate", accelerate.__version__, "| deepspeed", deepspeed.__version__,
      "| numpy", numpy.__version__, "| wandb", wandb.__version__)
EOF

# ---- accelerate config for this run only (your default config is not changed)
CLIP_LINE=""
[ "$GRAD_CLIP" != none ] && CLIP_LINE="  gradient_clipping: $GRAD_CLIP"
cat > "$ENV_DIR/accelerate_config.yaml" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 1
$CLIP_LINE
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: $ZERO_STAGE
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
sed -i '/^$/d' "$ENV_DIR/accelerate_config.yaml"
echo "wrote $ENV_DIR/accelerate_config.yaml (DeepSpeed ZeRO stage $ZERO_STAGE, bf16, 4 processes, clipping: $GRAD_CLIP)"
echo "environment ready in $ENV_DIR; now run: bash $HERE/submit.sh"
