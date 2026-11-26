#!/bin/bash
# Build (or rebuild) the zip from whatever is in a run folder, or show the state of each arm.  Safe to run at any time, any number
# of times, while jobs are running.   bash collect.sh hpc-outputs/run-<id>   |   bash collect.sh hpc-outputs/run-<id> --status
# It is quick (seconds, plus a minute or so for the pictures), so the login node is fine; add --no-figs to skip the pictures.
[ $# -ge 1 ] || { echo "usage: bash collect.sh hpc-outputs/run-<id> [--status] [--no-figs]"; exit 1; }
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
ARGS=("$@"); RUN=$(readlink -f "${ARGS[0]}"); ARGS[0]="$RUN"
LOAD_CUDA=0
source "$HERE/env.sh"
cd "$HERE" && python collect_run.py "${ARGS[@]}"
