#!/bin/bash
P=/tmp/claude-0/-home-claude/eb8a2a47-60d6-53a3-8d73-a8455129e63a/scratchpad/tvenv/bin/python
cd /home/claude/permutations-redux/claude-analysis/decompile
$P -u prune_stage1.py general 0 lin 100 runs/general_lin_l0.json 4 6 200 > runs/general_lin_l0.log 2>&1
for lam in 0.01 0.003 0.03; do
  $P -u prune_stage1.py general $lam lin 400 runs/general_lin_l$lam.json 4 6 200 > runs/general_lin_l$lam.log 2>&1
done
echo done > runs/general_sweep_done
