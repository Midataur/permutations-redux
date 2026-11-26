"""Compare the environment of this run with the original torn-bigger-13 run (its wandb requirements.txt).

Checks: versions of the packages that can change the numbers, and the accelerate default config (4 processes,
DeepSpeed, bf16, as logged for the original run).  Prints what it finds; exit code 1 if anything differs.

Usage: python check_env.py ENV_DIR ORIGINAL_REQUIREMENTS
"""
import os, sys, re

env, orig = sys.argv[1], sys.argv[2]
KEY = ["torch", "accelerate", "deepspeed", "numpy", "safetensors", "scipy", "wandb"]


def versions(path):
    v = {}
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line.strip())
            if m:
                v[m.group(1).lower().replace("_", "-")] = m.group(2)
    return v


o, c = versions(orig), versions(os.path.join(env, "pip_freeze.txt"))
bad = []
print("package       original run   this environment")
for k in KEY:
    a, b = o.get(k, "-"), c.get(k, "missing")
    flag = "" if a == b else "   <-- differs"
    if a != b and k in ("torch", "accelerate", "deepspeed"):
        bad.append(k)
    print(f"  {k:12s} {a:14s} {b}{flag}")

cfg = {}
p = os.path.join(env, "accelerate_default_config.yaml")
if os.path.exists(p):
    for line in open(p):
        m = re.match(r"^\s*([A-Za-z_]+):\s*(.*?)\s*$", line)
        if m:
            cfg[m.group(1)] = m.group(2).strip("'\"")
    print("accelerate default config:", {k: cfg.get(k) for k in
          ("distributed_type", "num_processes", "mixed_precision", "zero_stage", "gradient_accumulation_steps")})
    want = dict(distributed_type="DEEPSPEED", num_processes="4", mixed_precision="bf16")
    for k, v in want.items():
        if cfg.get(k) != v:
            bad.append(f"accelerate {k}={cfg.get(k)} (original run: {v})")
    if cfg.get("gradient_accumulation_steps", "1") not in ("1", ""):
        bad.append("accelerate gradient_accumulation_steps is not 1")
else:
    bad.append("no accelerate default_config.yaml found (the original run used `accelerate launch` with it)")

if bad:
    print("DIFFERENCES FROM THE ORIGINAL RUN:")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print("environment matches the original run on the checked items")
