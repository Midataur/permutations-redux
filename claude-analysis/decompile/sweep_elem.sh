#!/bin/bash
P=/tmp/claude-0/-home-claude/eb8a2a47-60d6-53a3-8d73-a8455129e63a/scratchpad/tvenv/bin/python
cd /home/claude/permutations-redux/claude-analysis/decompile
until [ -f runs/general_sweep_done ]; do sleep 30; done
$P -u prune_stage1.py elem 0 lin 100 runs/elem_lin_l0.json 3 4 200 > runs/elem_lin_l0.log 2>&1
$P -u prune_stage1.py elem 0 real 50 runs/elem_real_l0.json 3 4 0 > runs/elem_real_l0.log 2>&1
echo done > runs/elem_l0_done
