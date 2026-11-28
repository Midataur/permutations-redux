#!/bin/bash
P=/tmp/claude-0/-home-claude/eb8a2a47-60d6-53a3-8d73-a8455129e63a/scratchpad/tvenv/bin/python
cd /home/claude/permutations-redux/claude-analysis/decompile
for spec in "g_m10 0.0003 400" "g_plant10 0.0003 400" "g_m10 0.001 400" "g_plant10 0.001 400" "g_m10 0 100" "g_plant10 0 100"; do
  set -- $spec
  $P -u prune_stage1.py $1 $2 lin $3 runs/$1_lin_l$2.json 4 6 200 > runs/$1_lin_l$2.log 2>&1
done
echo done > runs/retrained_done
