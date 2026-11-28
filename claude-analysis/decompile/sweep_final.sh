#!/bin/bash
P=/tmp/claude-0/-home-claude/eb8a2a47-60d6-53a3-8d73-a8455129e63a/scratchpad/tvenv/bin/python
cd /home/claude/permutations-redux/claude-analysis/decompile
until [ -f runs/last_done ]; do sleep 30; done
$P -u prune_stage1.py elem 0.003 real 400 runs/elem_real_l0.003.json 3 4 0 > runs/elem_real_l0.003.log 2>&1
echo done > runs/final_done
