# Sourced by job_arm.slurm and collect.sh: modules and Python environment.  (Not run on its own.)
# Set PY_MODULES="" if your venv does not need modules, or to other modules; VENV=/path/to/venv to pick the environment.
PY_MODULES=${PY_MODULES-"GCCcore/13.3.0 XZ/5.4.5 Python/3.14.2"}
if type module >/dev/null 2>&1; then
    [ "${LOAD_CUDA:-1}" = "1" ] && module load CUDA/12.4.1
    for m in $PY_MODULES; do module load "$m" || echo "WARNING: could not load module $m"; done
else
    echo "(no 'module' command here: using the python on PATH)"
fi
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}" || exit 1
for cand in "$VENV" ./venv ../scaling-generator/venv ../venv; do
    if [ -n "$cand" ] && [ -f "$cand/bin/activate" ]; then source "$cand/bin/activate"; echo "using venv $cand"; break; fi
done
