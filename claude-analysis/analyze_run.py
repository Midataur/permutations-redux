"""
Run the mechanism tests (sections 3.9 and 3.12 and the planted-chain accuracy sweep) on the checkpoints of a finished hpc-experiment run.

    python analyze_run.py hpc-run-<id>.zip hpc-run-<id>-extras.zip  [--out analysis-<id>] [--arms g_m10,e_base] [--patching] [--jobs 2]
    python analyze_run.py DIR_WITH_UNZIPPED_FILES ...

It takes the LAST checkpoint of each arm (highest step; for an arm stopped at the deadline that is the step where it stopped) from
<root>-extras/<arm>/ckpt/step_*.safetensors, runs each test with the checkpoint, the block count and an output tag set through
ckpt_env.py, and writes one folder per arm plus summary.md (compare_arms.py).  Finished jobs are skipped, so it can be re-run.
Stages (cheap first, so partial results arrive early):  1 accuracy, attention, natural-word control, probes   2 head ablation
(zeroing)   3 head ablation (mean replacement, needs stage 2)   4 activation patching (only with --patching; about 15 minutes per arm).
"""
import argparse, glob, json, os, re, subprocess, sys, zipfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "hpc-experiment"))
try:
    from exp.arms import ARMS
except Exception:                                           # the arm table is only needed for the block count and kind
    ARMS = {"g_m10": dict(kind="general", n_blocks=5), "g_plant10": dict(kind="general", n_blocks=5),
            "g_plant10_b8": dict(kind="general", n_blocks=8), "e_base": dict(kind="elementary", n_blocks=5)}
# hop counts for the planted-word tests: the control stops being right near 16, the others are tested up to 24
LS = {"g_m10": "8,12,16", "g_plant10": "8,16,24", "g_plant10_b8": "8,16,24"}

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--out", default="analysis-run")
ap.add_argument("--arms", default="")
ap.add_argument("--patching", action="store_true")
ap.add_argument("--jobs", type=int, default=2)
ap.add_argument("--quick", action="store_true", help="a quarter of the words everywhere (about 20 minutes in total)")
a = ap.parse_args()
out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
work = os.path.join(out, "_unzipped")

for p in a.paths:                                           # unzip (only what is needed)
    if os.path.isdir(p):
        work = p if len(a.paths) == 1 else work
        continue
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            if n.endswith((".safetensors", "config.json", "summary.json", "status.json")):
                z.extract(n, work)

ck = {}
for f in glob.glob(os.path.join(work, "**", "ckpt", "step_*.safetensors"), recursive=True):
    arm = os.path.basename(os.path.dirname(os.path.dirname(f)))
    step = int(re.search(r"step_(\d+)", f).group(1))
    if arm in ARMS and (arm not in ck or step > ck[arm][0]):
        ck[arm] = (step, f)
want = [x for x in a.arms.split(",") if x] or list(ck)
missing = [x for x in want if x not in ck]
if missing:
    print("no checkpoint found for:", missing, "(did you pass the -extras zip?)")
arms = [x for x in want if x in ck]
print("checkpoints:", {x: (ck[x][0], os.path.basename(ck[x][1])) for x in arms}, flush=True)

q = 4 if a.quick else 1
def jobs_for(arm):
    kind = "general" if ARMS[arm]["kind"] == "general" else "elem"
    n = lambda x: str(max(10, x // q))
    J = {1: [], 2: [], 3: [], 4: []}
    J[1] += [("arm_accuracy.py", ["run_arm_accuracy.py", kind, n(300)], f"arm_accuracy_{kind}_{arm}_results.json")]
    J[1] += [("attention", ["run_attention.py", kind, n(300 if kind == "general" else 200)], f"attention_{kind}_{arm}_results.json")]
    J[1] += [("content", ["run_attention_content.py", kind, n(200 if kind == "general" else 150)], f"attention_content_{kind}_{arm}_results.json")]
    J[1] += [("probes", ["run_input_probes.py" if kind == "general" else "run_input_probes_elem.py", n(700 if kind == "general" else 450)],
              f"input_probe_results_{arm}.json" if kind == "general" else f"input_probe_elem_results_{arm}.json")]
    J[2] += [("ablation", ["run_head_ablation.py", kind, n(200 if kind == "general" else 100)], f"head_ablation_{kind}_{arm}_results.json")]
    J[3] += [("ablation_mean", ["run_head_ablation_mean.py", kind, n(200 if kind == "general" else 100), n(150)], f"head_ablation_mean_{kind}_{arm}_results.json")]
    if a.patching:
        J[4] += [("patching", ["run_patching.py" if kind == "general" else "run_patching_elem.py", n(200)],
                  f"patching_results_{arm}.json" if kind == "general" else f"patching_elem_results_{arm}.json")]
    return J

def run(arm, name, argv, result):
    d = os.path.join(out, arm)
    if os.path.exists(os.path.join(d, result)):
        return f"{arm}/{name}: already done"
    env = dict(os.environ, ATTN_CKPT=ck[arm][1], ATTN_BLOCKS=str(ARMS[arm]["n_blocks"]), ATTN_TAG=arm,
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if arm in LS:
        env["ATTN_LS"] = LS[arm]
    with open(os.path.join(d, name + ".log"), "w") as lf:
        r = subprocess.run([sys.executable, os.path.join(HERE, argv[0])] + argv[1:], cwd=d, env=env, stdout=lf, stderr=subprocess.STDOUT)
    ok = r.returncode == 0 and os.path.exists(os.path.join(d, result))
    return f"{arm}/{name}: {'ok' if ok else 'FAILED (see ' + name + '.log)'}"

for arm in arms:
    os.makedirs(os.path.join(out, arm), exist_ok=True)
    json.dump(dict(arm=arm, kind="general" if ARMS[arm]["kind"] == "general" else "elementary", blocks=ARMS[arm]["n_blocks"],
                   step=ck[arm][0], checkpoint=os.path.basename(ck[arm][1])), open(os.path.join(out, arm, "run_info.json"), "w"))
allj = {arm: jobs_for(arm) for arm in arms}
for stage in (1, 2, 3, 4):
    todo = [(arm, *j) for arm in arms for j in allj[arm][stage]]
    if not todo:
        continue
    print(f"--- stage {stage}: {len(todo)} jobs", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for msg in ex.map(lambda t: run(*t), todo):
            print(msg, flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "compare_arms.py"), out], stdout=subprocess.DEVNULL)
print(f"done. See {out}/summary.md")
