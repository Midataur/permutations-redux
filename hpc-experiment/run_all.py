#!/usr/bin/env python
"""One-job orchestrator: build eval sets -> calibrate throughput -> train all arms in parallel (one per GPU) ->
collect everything into a zip.   Called by run.slurm; can also be run by hand:  python run_all.py --help"""
import argparse, json, os, platform, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp.arms import ARMS, DEFAULT_ARMS, SPEC          # noqa: E402

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


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=os.environ.get("ARMS", " ".join(DEFAULT_ARMS)))
    ap.add_argument("--out-base", default=os.environ.get("OUT_BASE", "hpc-outputs"))
    ap.add_argument("--train-hours", type=float, default=env_float("TRAIN_HOURS", 8.0), help="target training time per arm")
    ap.add_argument("--wall-hours", type=float, default=env_float("WALL_HOURS", 10.0), help="the sbatch --time; arms stop 20 min before it")
    ap.add_argument("--steps-g", type=int, default=int(env_float("STEPS_G", 0)), help="override calibrated step count (general arms)")
    ap.add_argument("--steps-e", type=int, default=int(env_float("STEPS_E", 0)), help="override calibrated step count (elementary arm)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true", help="smoke-test sizes (CPU friendly)")
    ap.add_argument("--calib-seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=int(env_float("SEED", 0)))
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--resume-dir", default=os.environ.get("RESUME_DIR", ""), help="continue a previous run directory")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    arms = a.arms.replace(",", " ").split()
    for x in arms:
        assert x in ARMS, f"unknown arm {x}; choose from {list(ARMS)}"

    run_id = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%d-%H%M%S")
    if a.resume_dir:
        out = os.path.abspath(a.resume_dir); run_id = os.path.basename(out).replace("run-", "")
    else:
        out = os.path.abspath(os.path.join(a.out_base, f"run-{run_id}"))
    os.makedirs(out, exist_ok=True)
    LOG = os.path.join(out, "orchestrator.log")
    say(f"run {run_id} -> {out}; arms {arms}")
    stop_path = os.path.join(out, "STOP")
    if os.path.exists(stop_path):
        os.remove(stop_path)

    import numpy as np
    import torch
    ngpu = torch.cuda.device_count() if a.device == "cuda" else 0
    if a.device == "cuda" and ngpu == 0:
        say("ERROR: --device cuda but no GPU is visible"); sys.exit(2)
    env_info = dict(host=platform.node(), python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, n_gpu=ngpu,
                    gpus=[torch.cuda.get_device_name(i) for i in range(ngpu)],
                    slurm={k: v for k, v in os.environ.items() if k.startswith("SLURM_") and "JOB" in k}, args=vars(a))
    try:
        env_info["git_rev"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    json.dump(env_info, open(os.path.join(out, "env.json"), "w"), indent=1)
    say(f"torch {torch.__version__}, {ngpu} GPU(s): {env_info['gpus']}")

    # ---------------------------------------------------------------- 1. evaluation sets (shared by all arms)
    ev_path = os.path.join(out, "evalsets.npz")
    if not os.path.exists(ev_path):
        from exp import evalsets
        from exp.datagen import ElemSampler
        t = time.time(); data = {}
        n_nat, per_cell = (200, 20) if a.tiny else (2000, 300)
        if any(ARMS[x]["kind"] == "general" for x in arms):
            data.update(evalsets.build_general(n_nat=n_nat, per_cell=per_cell))
        if any(ARMS[x]["kind"] == "elementary" for x in arms):
            tl = ElemSampler("cpu", 99).words(n_nat).numpy()
            data.update(evalsets.build_elem(n_nat=n_nat, per_cell=per_cell, train_like_words=tl))
        np.savez_compressed(ev_path, **data)
        say(f"evaluation sets built in {time.time() - t:.0f}s: {evalsets.set_names(data)}")

    def launch(extra, arm, gpu):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if a.device == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = str(gpu % max(ngpu, 1))
        cmd = [sys.executable, "-m", "exp.train", "--arm", arm, "--out", out, "--evalsets", ev_path, "--seed", str(a.seed),
               "--device", a.device, "--batch", str(a.batch)] + (["--tiny"] if a.tiny else []) + extra
        os.makedirs(os.path.join(out, arm), exist_ok=True)
        lf = open(os.path.join(out, arm, "train.log"), "a")
        return subprocess.Popen(cmd, cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)

    # ---------------------------------------------------------------- 2. calibration -> number of steps
    plan_path = os.path.join(out, "plan.json")
    if os.path.exists(plan_path):
        plan = json.load(open(plan_path)); say("re-using plan.json")
    else:
        calib = {}
        say(f"calibrating throughput for {a.calib_seconds:.0f}s (all arms at once, so contention is included)")
        procs = {x: launch(["--steps", "1", "--calibrate", str(a.calib_seconds)], x, i) for i, x in enumerate(arms)}
        for x, p in procs.items():
            p.wait()
            cp = os.path.join(out, x, "calib.json")
            if p.returncode == 0 and os.path.exists(cp):
                calib[x] = json.load(open(cp))
            else:
                say(f"ERROR: calibration of {x} failed (exit {p.returncode}); see {out}/{x}/train.log")
        if len(calib) < len(arms):
            say("aborting: an arm cannot even run 15 training steps (this is the early self-check doing its job).")
            from exp.collect import collect
            zp, _, _ = collect(out, run_id, arms, make_figs=False)
            say(f"the logs are in {zp}; upload it and I can fix the problem")
            sys.exit(3)
        budget = a.train_hours * 3600.0
        def steps_for(x):
            c = calib[x]; n_snap = 62
            usable = budget - n_snap * c["eval_seconds"] - 90
            return max(500, int(usable * c["steps_per_sec"] * 0.93))
        sg = [steps_for(x) for x in arms if ARMS[x]["kind"] == "general"]
        se = [steps_for(x) for x in arms if ARMS[x]["kind"] == "elementary"]
        plan = dict(steps_general=a.steps_g or (min(sg) // 500 * 500 if sg else 0),
                    steps_elementary=a.steps_e or (min(se) // 500 * 500 if se else 0),
                    calibration=calib, train_hours=a.train_hours, wall_hours=a.wall_hours)
        json.dump(plan, open(plan_path, "w"), indent=1)
    say("plan: " + json.dumps({k: v for k, v in plan.items() if k != "calibration"}))
    for x, c in plan.get("calibration", {}).items():
        say(f"  {x}: {c['steps_per_sec']:.2f} steps/s, evaluation {c['eval_seconds']:.1f}s, {c['params']/1e6:.1f}M params")

    # ---------------------------------------------------------------- 3. train every arm
    deadline = T_START + a.wall_hours * 3600 - 20 * 60
    stop_requested = {"v": False}

    def on_signal(signum, frame):
        say(f"signal {signum}: asking the arms to stop and collecting what exists")
        stop_requested["v"] = True
        open(stop_path, "w").write("stop")
    for sg_ in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sg_, on_signal)

    procs = {}
    for i, x in enumerate(arms):
        steps = plan["steps_general"] if ARMS[x]["kind"] == "general" else plan["steps_elementary"]
        procs[x] = launch(["--steps", str(steps), "--deadline", str(deadline)] + (["--resume"] if a.resume_dir else []), x, i)
        say(f"started {x} on GPU {i % max(ngpu, 1)}: {steps} steps")
    last_report = 0.0
    while any(p.poll() is None for p in procs.values()):
        time.sleep(5)
        if time.time() - last_report > 300:
            last_report = time.time()
            for x in arms:
                try:
                    st = json.load(open(os.path.join(out, x, "status.json")))
                    say(f"  {x}: {st['state']} step {st['step']}/{st['total']} ({st['elapsed_s']/3600:.2f} h)")
                except Exception:
                    say(f"  {x}: no status yet")
        if time.time() > deadline + 600 and not stop_requested["v"]:
            on_signal(0, None)
        if stop_requested["v"] and time.time() > deadline + 900:
            for p in procs.values():
                if p.poll() is None: p.kill()
    codes = {x: p.returncode for x, p in procs.items()}
    say(f"all arms finished: exit codes {codes}")

    # ---------------------------------------------------------------- 4. collect
    from exp.collect import collect
    main_zip, ex_zip, dt = collect(out, run_id, arms, make_figs=not a.no_figs)
    say(f"collected in {dt:.0f}s")
    bar = "=" * 78
    say(bar)
    say(f"UPLOAD THIS FILE:   {main_zip}   ({os.path.getsize(main_zip)/1e6:.1f} MB)")
    if ex_zip:
        say(f"optional (checkpoints, token embeddings):   {ex_zip}   ({os.path.getsize(ex_zip)/1e6:.1f} MB)")
    say(bar)
    sys.exit(0 if all(c == 0 for c in codes.values()) else 1)


if __name__ == "__main__":
    main()
