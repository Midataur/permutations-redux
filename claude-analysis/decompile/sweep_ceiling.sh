#!/bin/bash
# Unpruned (lambda = 0, gates held open) linear-LayerNorm fits of the two retrained models (section 3.26); the two run in parallel, about 1.5 h each on 1 CPU thread.
P=/tmp/claude-0/-home-claude/eb8a2a47-60d6-53a3-8d73-a8455129e63a/scratchpad/tvenv/bin/python
cd /home/claude/permutations-redux/claude-analysis/decompile
for m in g_plant10 g_m10; do
  OMP_NUM_THREADS=1 $P -u prune_stage1.py $m 0 lin 2500 runs/${m}_lin_l0_long.json 4 6 200 > runs/${m}_lin_l0_long.log 2>&1 &
done
wait
echo done > runs/ceiling_done
