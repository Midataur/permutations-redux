"""The one-GPU-per-arm jobs (run_arm.py) must not depend on each other or on the order they start and finish in.
Tiny model, CPU.  (1) three arms started in a scrambled order, overlapping in time, each refreshing the combined zip when it
finishes: the zip must end up with all three, no temp files or locks left.  (2) an arm stopped through its own STOP file, then
resumed: it must finish with the original number of steps, and another arm must be unaffected by the stop."""
import glob, json, os, subprocess, sys, tempfile, time, zipfile

HERE = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
PY = sys.executable
base = tempfile.mkdtemp(prefix="hpc-indep-")
run = os.path.join(base, "hpc-outputs", "run-t")
os.makedirs(run)


def start(arm, steps, extra=()):
    cmd = [PY, "run_arm.py", "--arm", arm, "--run-dir", run, "--steps", str(steps), "--tiny", "--device", "cpu", "--calib-seconds", "2",
           "--no-figs", "--batch", "32"] + list(extra)
    return subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def state(arm):
    try: return json.load(open(os.path.join(run, arm, "status.json")))
    except Exception: return {}


# ---- (1) scrambled, overlapping starts
procs = {}
for arm, delay in (("e_base", 0), ("g_plant10_b8", 3), ("g_m10", 6)):
    time.sleep(delay if procs else 0); procs[arm] = start(arm, 150)
    print("started", arm, flush=True)
outs = {a: p.communicate()[0] for a, p in procs.items()}
for a, p in procs.items():
    assert p.returncode == 0, f"{a} exit {p.returncode}\n{outs[a][-3000:]}"
for a in procs:
    assert state(a).get("state") == "done" and state(a)["step"] == 150, (a, state(a))
zip_path = os.path.join(base, "hpc-outputs", "hpc-run-t.zip")
assert os.path.exists(zip_path), "no combined zip"
names = zipfile.ZipFile(zip_path).namelist()
for a in procs:
    assert any(n.endswith(f"/{a}/metrics.jsonl") for n in names), f"{a} missing from the combined zip"
    assert any(n.endswith(f"/{a}/job.log") for n in names) and any(n.endswith(f"/{a}/env.json") for n in names)
assert not glob.glob(os.path.join(base, "hpc-outputs", "*.tmp")) and not os.path.exists(os.path.join(run, ".collect.lock")), "leftover temp file or lock"
assert os.path.exists(os.path.join(run, "evalsets_general.npz")) and os.path.exists(os.path.join(run, "evalsets_elementary.npz"))
print("(1) ok: three independent arms, scrambled order, one complete zip:", len(names), "files", flush=True)

# ---- (2) per-arm STOP and resume (an arm with more steps, so that it is still running when we stop it)
p = start("g_plant5", 4000, ["--calib-seconds", "0"])
t0 = time.time()
while state("g_plant5").get("step", 0) < 100 and time.time() - t0 < 300 and p.poll() is None:
    time.sleep(1)
open(os.path.join(run, "g_plant5", "STOP"), "w").write("stop")
out = p.communicate(timeout=300)[0]
st = state("g_plant5")
assert st.get("state") == "truncated" and 0 < st["step"] < 4000, (st, out[-2000:])
assert state("g_m10").get("state") == "done", "stopping one arm must not touch another"
p2 = start("g_plant5", 999, ["--calib-seconds", "0", "--resume"])        # wrong steps on purpose: the original 4000 must win
out2 = p2.communicate(timeout=900)[0]
assert p2.returncode == 0, out2[-3000:]
st2 = state("g_plant5")
assert st2.get("state") == "done" and st2["step"] == 4000 and st2["total"] == 4000, st2
assert "using the original 4000 steps" in out2
print("(2) ok: stopped at step", st["step"], "then resumed to 4000", flush=True)
subprocess.run([PY, "collect_run.py", run, "--status"], cwd=HERE, check=True)
print("ALL OK")
