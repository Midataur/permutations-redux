"""Which checkpoint the analysis scripts use.  Defaults reproduce the original runs (torn-bigger-13 / elem-partitioned-long-boosted-1).
Set these environment variables to run the same script on another checkpoint, e.g. one trained by ../hpc-experiment:

    ATTN_CKPT=/path/to/step_00200000.safetensors   the checkpoint (repo format, as written by hpc-experiment)
    ATTN_BLOCKS=8                                   number of blocks (default 5)
    ATTN_TAG=g_plant10_b8                           put in every output file name:  attention_general_<TAG>_results.json
    ATTN_LS=8,16,24                                 hop counts for the planted-word scripts (default: 8,12,16 general; 12,40,80 elementary)
"""
import os

CKPT = os.environ.get("ATTN_CKPT") or None
BLOCKS = int(os.environ.get("ATTN_BLOCKS", "5"))
TAG = os.environ.get("ATTN_TAG", "")
SUF = f"_{TAG}" if TAG else ""
LS = tuple(int(x) for x in os.environ["ATTN_LS"].split(",")) if os.environ.get("ATTN_LS") else None
