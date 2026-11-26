#!/bin/bash
# CPU-only checks (needs torch, numpy, safetensors; matplotlib optional).  About 6 minutes.
set -e
cd "$(dirname "$0")/.."
python tests/test_equivalence.py general
python tests/test_equivalence.py elementary
python tests/test_datagen.py
python tests/test_independent_jobs.py
# end-to-end smoke test: tiny model, 4 arms, 120 steps, CPU
python run_all.py --tiny --device cpu --out-base /tmp/hpc-smoke --steps-g 120 --steps-e 120 --calib-seconds 3 --train-hours 0.05 --wall-hours 1
