#!/bin/bash
# ONE command: submits one single-GPU job per arm.   cd hpc-experiment && bash submit.sh
#
# The jobs are independent of each other.  They can start and finish in any order, on any nodes, at any time: each builds what
# it needs, trains for the same fixed number of steps as the other arms of its kind, and then refreshes the combined zip (so the
# last job to finish leaves the complete one).  Nothing waits for anything else.
#
#   Output:   hpc-outputs/hpc-run-<RUN_ID>.zip      (upload this; it is rewritten each time an arm finishes)
#             hpc-outputs/hpc-run-<RUN_ID>-extras.zip   (optional: checkpoints, token embeddings)
#   Look:     bash collect.sh hpc-outputs/run-<RUN_ID> --status
#   Rebuild the zip by hand at any time:   bash collect.sh hpc-outputs/run-<RUN_ID>
#
# Settings (environment variables):
#   ARMS="g_m10 g_plant10 g_plant10_b8 e_base"   which arms (one job each).  Extras: g_plant10_b3 g_plant5 g_m5
#   STEPS_G=200000   optimiser steps for every general arm (the same for all of them, so they can be compared)
#   STEPS_E=110000   optimiser steps for the elementary arm
#   WALL_HOURS=10    time limit of each job; training stops cleanly 20 minutes before it
#   BATCH=1024       words per step
#   RESUME_DIR=hpc-outputs/run-<id>   continue the arms in ARMS from their last resume.pt (same steps as before)
#   RUN_ID=...       name of the run folder (default: the date and time)
set -e
cd "$(dirname "$(readlink -f "$0")")"

ARMS=${ARMS:-"g_m10 g_plant10 g_plant10_b8 e_base"}
STEPS_G=${STEPS_G:-200000}
STEPS_E=${STEPS_E:-110000}
WALL_HOURS=${WALL_HOURS:-10}
export WALL_HOURS BATCH SEED PY_MODULES VENV

if [ -n "$RESUME_DIR" ]; then
    RUN_DIR=$(readlink -f "$RESUME_DIR"); [ -d "$RUN_DIR" ] || { echo "RESUME_DIR $RESUME_DIR does not exist"; exit 1; }
    RESUME=1
else
    RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
    RUN_DIR=$(readlink -f hpc-outputs)/run-$RUN_ID
    RESUME=""
fi
RUN_ID=$(basename "$RUN_DIR" | sed 's/^run-//')
mkdir -p "$RUN_DIR" "$(dirname "$RUN_DIR")/logs"
MINUTES=$(awk "BEGIN{printf \"%d\", $WALL_HOURS*60}")

echo "run folder: $RUN_DIR"
for arm in $ARMS; do
    case "$arm" in
        g_*) STEPS=$STEPS_G ;;
        e_*) STEPS=$STEPS_E ;;
        *) echo "unknown arm '$arm' (names start with g_ or e_; see exp/arms.py)"; exit 1 ;;
    esac
    EXPORT="ALL,ARM=$arm,RUN_DIR=$RUN_DIR,STEPS=$STEPS,WALL_HOURS=$WALL_HOURS"
    [ -n "$RESUME" ] && EXPORT="$EXPORT,RESUME=1"
    JID=$(sbatch --parsable --job-name="perm-$arm" --time="$MINUTES" \
          --output="$(dirname "$RUN_DIR")/logs/%x-%j.out" --export="$EXPORT" job_arm.slurm)
    printf "  submitted %-14s job %s   (%s steps, 1 GPU)\n" "$arm" "$JID" "$STEPS"
done
echo
echo "The jobs are independent and can run in any order.  When they are all done, upload:"
echo "    $(dirname "$RUN_DIR")/hpc-run-$RUN_ID.zip"
echo "State of the arms at any time:   bash collect.sh $RUN_DIR --status"
