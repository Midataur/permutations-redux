"""Locations of the trained checkpoints, relative to this folder (claude-analysis/ sits at the repo root).
Override with the TORN_MODEL environment variable if the repo is laid out differently."""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
MODEL_TORN = os.environ.get(
    "TORN_MODEL",
    str(_REPO / "scaling-generator" / "trained-models" / "torn-bigger-13" / "model.safetensors"))
