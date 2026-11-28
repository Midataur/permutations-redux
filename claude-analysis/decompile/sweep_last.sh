#!/bin/bash
P=/tmp/claude-0/-home-claude/eb8a2a47-60d6-53a3-8d73-a8455129e63a/scratchpad/tvenv/bin/python
cd /home/claude/permutations-redux/claude-analysis/decompile
until [ -f runs/next_done ]; do sleep 30; done
$P -u prune_stage1.py general 0.0003 lin 400 runs/general_lin_l0.0003.json 4 6 200 > runs/general_lin_l0.0003.log 2>&1
echo done > runs/last_done
