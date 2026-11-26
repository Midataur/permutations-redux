#!/usr/bin/env python
"""Bundle whatever exists in a run directory into hpc-outputs/hpc-run-<id>.zip (+ optional -extras.zip).

Order-agnostic and idempotent: it looks at the arm folders that are there, whatever state they are in (done, running, stopped
at the deadline, failed), so it can be run at any time, by any job, any number of times.  Every arm job (run_arm.py) calls it
when it finishes, so the zip is refreshed by each finishing job and the last one to finish leaves the complete one.  Calls are
serialised by a lock directory, and the zip is replaced atomically, so two jobs finishing at the same moment cannot leave a
half-written or older zip behind.

    python collect_run.py hpc-outputs/run-<id>            # build the zips
    python collect_run.py hpc-outputs/run-<id> --status   # only print the state of each arm
"""
import argparse, contextlib, json, os, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp.arms import ARMS                                   # noqa: E402


def arms_present(run_dir):
    found = [d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)) and d in ARMS
             and (os.path.isfile(os.path.join(run_dir, d, "config.json")) or os.path.isfile(os.path.join(run_dir, d, "status.json"))
                  or os.path.isfile(os.path.join(run_dir, d, "train.log")))]
    order = list(ARMS)
    return sorted(found, key=order.index)


def arm_state(run_dir, arm):
    try:
        st = json.load(open(os.path.join(run_dir, arm, "status.json")))
        return f"{st.get('state', '?')} step {st.get('step', '?')}/{st.get('total', '?')} ({st.get('elapsed_s', 0) / 3600:.2f} h)"
    except Exception:
        return "no status yet"


@contextlib.contextmanager
def lock(path, timeout=1800, stale=900):
    """mkdir is atomic on shared file systems (flock is not, on some).  A lock older than `stale` seconds is taken to be left
    over from a killed job.  After `timeout` we go ahead without it: the atomic rename still keeps every zip whole."""
    t0, held = time.time(), False
    while True:
        try:
            os.mkdir(path); held = True; break
        except FileExistsError:
            try: age = time.time() - os.path.getmtime(path)
            except OSError: age = 0
            if age > stale:
                shutil.rmtree(path, ignore_errors=True); continue
            if time.time() - t0 > timeout:
                break
            time.sleep(3)
    try:
        yield
    finally:
        if held:
            shutil.rmtree(path, ignore_errors=True)


def collect_all(run_dir, make_figs=True, log=print):
    out = os.path.abspath(run_dir)
    run_id = os.path.basename(out).replace("run-", "")
    with lock(os.path.join(out, ".collect.lock")):
        arms = arms_present(out)
        if not arms:
            log(f"collect: no arm folders in {out}"); return None, None
        from exp.collect import collect
        main_zip, ex_zip, dt = collect(out, run_id, arms, make_figs=make_figs)
        states = {a: arm_state(out, a) for a in arms}
        log(f"collect: {len(arms)} arm(s) in {dt:.0f}s: " + "; ".join(f"{a}: {s}" for a, s in states.items()))
        log(f"UPLOAD THIS FILE:   {main_zip}   ({os.path.getsize(main_zip) / 1e6:.1f} MB)   (rewritten each time an arm finishes; the last one to finish leaves the complete zip)")
        if ex_zip:
            log(f"optional (checkpoints, token embeddings):   {ex_zip}   ({os.path.getsize(ex_zip) / 1e6:.1f} MB)")
        return main_zip, ex_zip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--status", action="store_true", help="only print the state of each arm")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    if a.status:
        for arm in arms_present(a.run_dir):
            print(f"{arm:14s} {arm_state(os.path.abspath(a.run_dir), arm)}")
        sys.exit(0)
    z, _ = collect_all(a.run_dir, make_figs=not a.no_figs)
    sys.exit(0 if z else 1)
