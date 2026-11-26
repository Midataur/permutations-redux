#!/usr/bin/env python
"""ONE arm as a self-contained job (one GPU).  Called by job_arm.slurm; can also be run by hand:  python run_arm.py --help

Nothing here depends on any other arm or on the order in which jobs start or finish:
  * the evaluation sets are built from fixed seeds into  <run>/evalsets_<kind>.npz  (whoever gets there first writes it, to a
    temporary name and renames it; a second job that builds it at the same time writes identical bytes);
  * the number of steps is given (the same value for every arm of a kind, chosen when the jobs are submitted), so all general arms
    train for the same number of steps without having to measure each other;
  * a short self-check/throughput run comes first (fails in the first minute if the arm cannot run, and tells you whether the steps
    fit in the time limit);
  * at the end the arm refreshes the combined zip with collect_run.py (locked, atomic), so the last job to finish leaves the
    complete zip, whatever the order was.
"""
import argparse, json, os, platform, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp.arms import ARMS                                   # noqa: E402

T_START = time.time()
LOG = None


def say(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def env_float(name, default):
    v = os.environ.get(name, "")
    return float(v) if v.strip() else default


def ensure_evalsets(out, kind, tiny):
    path = os.path.join(out, f"evalsets_{kind}.npz")
    if os.path.exists(path):
        return path
    import numpy as np
    from exp import evalsets
    from exp.datagen import ElemSampler
    t = time.time()
    n_nat, per_cell = (200, 20) if tiny else (2000, 300)
    if kind == "general":
        data = evalsets.build_general(n_nat=n_nat, per_cell=per_cell)
    else:
        tl = ElemSampler("cpu", 99).words(n_nat).numpy()
        data = evalsets.build_elem(n_nat=n_nat, per_cell=per_cell, train_like_words=tl)
    tmp = f"{path}.{os.getpid()}.tmp.npz"                  # numpy appends .npz unless the name already ends with it
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)
    say(f"evaluation sets ({kind}) built in {time.time() - t:.0f}s: {evalsets.set_names(data)}")
    return path


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--run-dir", required=True, help="shared run folder, e.g. hpc-outputs/run-20261126-1100 (created if missing)")
    ap.add_argument("--steps", type=int, required=True, help="optimiser steps (use the same for all arms of a kind)")
    ap.add_argument("--batch", type=int, default=int(env_float("BATCH", 1024)))
    ap.add_argument("--seed", type=int, default=int(env_float("SEED", 0)))
    ap.add_argument("--wall-hours", type=float, default=env_float("WALL_HOURS", 10.0), help="the sbatch --time; training stops 20 min before it")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true", help="smoke-test sizes (CPU friendly)")
    ap.add_argument("--calib-seconds", type=float, default=env_float("CALIB_SECONDS", 30.0))
    ap.add_argument("--resume", action="store_true", help="continue this arm from its last resume.pt (same steps as before)")
    ap.add_argument("--no-collect", action="store_true", help="do not refresh the combined zip at the end")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.run_dir)
    adir = os.path.join(out, a.arm)
    os.makedirs(adir, exist_ok=True)
    LOG = os.path.join(adir, "job.log")
    kind = ARMS[a.arm]["kind"]
    steps = a.steps
    if a.resume and os.path.exists(os.path.join(adir, "config.json")):
        prev = json.load(open(os.path.join(adir, "config.json")))["steps"]
        if prev != steps:
            say(f"resuming: using the original {prev} steps instead of {steps} (the learning-rate schedule depends on it)")
            steps = prev
    for stale in (os.path.join(adir, "STOP"),):
        if os.path.exists(stale):
            os.remove(stale)
    say(f"arm {a.arm} ({kind}) -> {adir}; {steps} steps x {a.batch} words = {steps * a.batch / 1e6:.0f}M words; wall limit {a.wall_hours} h")

    import torch
    ngpu = torch.cuda.device_count() if a.device == "cuda" else 0
    if a.device == "cuda" and ngpu == 0:
        say("ERROR: --device cuda but no GPU is visible"); sys.exit(2)
    env_info = dict(host=platform.node(), python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, n_gpu=ngpu,
                    gpus=[torch.cuda.get_device_name(i) for i in range(ngpu)],
                    slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_") and "JOB" in k}, args=vars(a), steps=steps)
    try:
        env_info["git_rev"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    json.dump(env_info, open(os.path.join(adir, "env.json"), "w"), indent=1)
    say(f"torch {torch.__version__}, {ngpu} GPU(s): {env_info['gpus']} on {env_info['host']}")

    ev_path = ensure_evalsets(out, kind, a.tiny)

    def launch(extra):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        cmd = [sys.executable, "-m", "exp.train", "--arm", a.arm, "--out", out, "--evalsets", ev_path, "--seed", str(a.seed),
               "--device", a.device, "--batch", str(a.batch)] + (["--tiny"] if a.tiny else []) + extra
        lf = open(os.path.join(adir, "train.log"), "a")
        return subprocess.Popen(cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)

    def finish(code):
        if not a.no_collect:
            try:
                from collect_run import collect_all
                collect_all(out, make_figs=not a.no_figs, log=say)
            except Exception as e:                          # collecting must never turn a good run into a failed one
                say(f"collect failed (run  python collect_run.py {out}  later): {e!r}")
        sys.exit(code)

    # ---------------------------------------------------------------- 1. self-check and throughput (this arm only)
    if a.calib_seconds > 0 and not a.resume:
        say(f"self-check: {a.calib_seconds:.0f}s of real training to measure speed")
        p = launch(["--steps", "1", "--calibrate", str(a.calib_seconds)])
        p.wait()
        cp = os.path.join(adir, "calib.json")
        if p.returncode != 0 or not os.path.exists(cp):
            say(f"ERROR: the self-check failed (exit {p.returncode}); see {adir}/train.log")
            finish(3)
        c = json.load(open(cp))
        hours = steps / c["steps_per_sec"] / 3600 + 62 * c["eval_seconds"] / 3600
        say(f"speed {c['steps_per_sec']:.2f} steps/s, one evaluation {c['eval_seconds']:.1f}s, {c['params'] / 1e6:.1f}M params: "
            f"{steps} steps need about {hours:.1f} h (including about 62 evaluations); available {a.wall_hours - 0.4:.1f} h")
        if hours > a.wall_hours - 0.4:
            say("WARNING: this will not finish inside the time limit. Training will stop cleanly at the deadline and can be "
                f"continued:  ARMS={a.arm} RESUME_DIR={out} bash submit.sh   (or lower STEPS_G / STEPS_E for the next submission)")
        words = steps * a.batch
        if words < 1.5 * 160e6 and not a.tiny:
            say(f"note: {words / 1e6:.0f}M words is under 1.5x the 160M the original runs needed to reach 99.9% train accuracy (README, 'How many steps')")

    # ---------------------------------------------------------------- 2. train
    deadline = T_START + a.wall_hours * 3600 - 20 * 60
    stop_path = os.path.join(adir, "STOP")
    stop_requested = {"v": False}

    def on_signal(signum, frame):
        say(f"signal {signum}: asking the training process to stop cleanly")
        stop_requested["v"] = True
        open(stop_path, "w").write("stop")
    for sg in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sg, on_signal)

    proc = launch(["--steps", str(steps), "--deadline", str(deadline)] + (["--resume"] if a.resume else []))
    say(f"training started (pid {proc.pid})")
    last_report = 0.0
    while proc.poll() is None:
        time.sleep(5)
        if time.time() - last_report > 600:
            last_report = time.time()
            try:
                st = json.load(open(os.path.join(adir, "status.json")))
                say(f"  {st['state']} step {st['step']}/{st['total']} ({st['elapsed_s'] / 3600:.2f} h)")
            except Exception:
                say("  no status yet")
        if time.time() > deadline + 600 and not stop_requested["v"]:
            on_signal(0, None)
        if stop_requested["v"] and time.time() > deadline + 900 and proc.poll() is None:
            proc.kill()
    say(f"training finished with exit code {proc.returncode}")
    finish(0 if proc.returncode == 0 else 1)


if __name__ == "__main__":
    main()
