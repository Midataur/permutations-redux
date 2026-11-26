"""Train ONE arm.   python -m exp.train --arm g_plant10 --out DIR --evalsets DIR/evalsets.npz --steps 100000

Writes into DIR/<arm>/ :  metrics.jsonl (one line per snapshot), pos/step_*.npy (position embeddings),
tok/step_*.npy (token embeddings, fewer snapshots), ckpt/*.safetensors (repo format, bfloat16), status.json, log.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

from .arms import ARMS, SPEC
from .model import Net, save_original_safetensors
from .datagen import GeneralSampler, ElemSampler, general_batch, elem_batch
from .evaluate import evaluate_all

BASE_LR, WARMUP, DECAY_START, FINAL_FRAC = 3e-4, 100, 0.6, 0.1


def lr_at(step, total):
    if step < WARMUP:
        return BASE_LR * (step + 1) / WARMUP
    f = step / total
    if f <= DECAY_START:
        return BASE_LR
    c = 0.5 * (1 + math.cos(math.pi * (f - DECAY_START) / (1 - DECAY_START)))
    return BASE_LR * (FINAL_FRAC + (1 - FINAL_FRAC) * c)


def snapshot_schedule(total, n_log=36, n_lin=24):
    steps = {0, total}
    steps |= set(int(round(x)) for x in np.geomspace(1, total, n_log))
    steps |= set(int(round(x)) for x in np.linspace(0, total, n_lin + 1))
    return sorted(s for s in steps if 0 <= s <= total)


def load_sets(path, kind):
    z = np.load(path)
    keys = [k for k in z.files]
    if kind == "general":
        keep = {"nat25", "nat10", "ladder_full", "ladder_in10", "ladder_in5"}
    else:
        keep = {"nat16", "train_like", "cells"}
    return {k: z[k] for k in keys if k.split("__")[0] in keep}


def log(msg, f=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--evalsets", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="small model for smoke tests")
    ap.add_argument("--calibrate", type=float, default=0.0, help="only measure steps/s for this many seconds")
    ap.add_argument("--deadline", type=float, default=0.0, help="unix time after which training stops gracefully")
    ap.add_argument("--ckpt-fracs", default="0.25,1.0")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    arm = ARMS[a.arm]; kind = arm["kind"]; spec = SPEC[kind]
    dev = torch.device(a.device)
    bf16 = (dev.type == "cuda") and not a.no_bf16
    adir = os.path.join(a.out, a.arm)
    for sub in ("pos", "tok", "ckpt"):
        os.makedirs(os.path.join(adir, sub), exist_ok=True)
    status_path = os.path.join(adir, "status.json")
    stop_paths = (os.path.join(a.out, "STOP"), os.path.join(adir, "STOP"))        # run-wide (run_all.py) or this arm only (run_arm.py)
    t0 = time.time()

    def status(state, **kw):
        d = dict(arm=a.arm, state=state, step=int(step), total=a.steps, elapsed_s=round(time.time() - t0, 1), **kw)
        with open(status_path + ".tmp", "w") as f:
            json.dump(d, f)
        os.replace(status_path + ".tmp", status_path)

    step = 0
    try:
        D, H = (48, 6) if a.tiny else (402, 6)
        torch.manual_seed(a.seed)                       # same initial weights for every arm of the same architecture
        model = Net(spec["vocab"], spec["n_ctx"], spec["input_len"], spec["n_out"], D=D, H=H, n_blocks=arm["n_blocks"])
        model = model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.05)
        dseed = a.seed * 1000 + 17 + sum(map(ord, a.arm))
        if kind == "general":
            sampler = GeneralSampler(dev, arm["dist"], dseed); mk = general_batch
        else:
            sampler = ElemSampler(dev, dseed); mk = elem_batch
        B = 32 if a.tiny else a.batch
        n_params = sum(p.numel() for p in model.parameters())
        log(f"{a.arm}: {kind}, {arm['n_blocks']} blocks, D={D}, params={n_params/1e6:.2f}M, batch={B}, steps={a.steps}, bf16={bf16}, dev={dev}")

        # ------------------------------------------------------------------ calibration only
        if a.calibrate > 0:
            sets = load_sets(a.evalsets, kind)
            def one():
                w, p = sampler.sample(B); inp, tgt = mk(w, p)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=bf16):
                    lg = model(inp)
                loss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), tgt.reshape(-1))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            for _ in range(15):
                one()
            if dev.type == "cuda": torch.cuda.synchronize()
            ts, n = time.time(), 0
            while time.time() - ts < a.calibrate:
                for _ in range(10):
                    one()
                n += 10
            if dev.type == "cuda": torch.cuda.synchronize()
            sps = n / (time.time() - ts)
            te = time.time(); evaluate_all(model, kind, sets, dev, bf16=bf16)
            if dev.type == "cuda": torch.cuda.synchronize()
            ev = time.time() - te
            json.dump(dict(arm=a.arm, steps_per_sec=sps, eval_seconds=ev, batch=B, params=n_params), open(os.path.join(adir, "calib.json"), "w"))
            log(f"calibration: {sps:.2f} steps/s, one evaluation {ev:.1f}s")
            return 0

        sets = load_sets(a.evalsets, kind)
        total = a.steps
        sched = snapshot_schedule(total)
        tok_steps = set([sched[i] for i in range(0, len(sched), max(1, len(sched) // 8))] + [0, total])
        ckpt_steps = {int(round(float(f) * total)) for f in a.ckpt_fracs.split(",")}
        json.dump(dict(arm=a.arm, desc=arm["desc"], kind=kind, dist=arm.get("dist"), n_blocks=arm["n_blocks"], D=D, H=H,
                       batch=B, steps=total, seed=a.seed, base_lr=BASE_LR, warmup=WARMUP, decay_start=DECAY_START,
                       final_lr_frac=FINAL_FRAC, weight_decay=0.05, bf16=bf16, params=n_params, snapshots=sched,
                       tok_snapshots=sorted(tok_steps), tiny=a.tiny), open(os.path.join(adir, "config.json"), "w"), indent=1)

        resume_path = os.path.join(adir, "resume.pt")
        if a.resume and os.path.exists(resume_path):
            ck = torch.load(resume_path, map_location=dev)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
            sampler.g.manual_seed(dseed + step)
            log(f"resumed from step {step}")
        metrics_path = os.path.join(adir, "metrics.jsonl")
        done_steps = set()
        if os.path.exists(metrics_path):
            for line in open(metrics_path):
                try: done_steps.add(json.loads(line)["step"])
                except Exception: pass

        run_loss = torch.zeros((), device=dev); run_acc = torch.zeros((), device=dev); run_n = 0
        words_per_step = B

        def snap(s, truncated=False):
            nonlocal run_loss, run_acc, run_n
            if s in done_steps:
                return
            tl = (run_loss / max(run_n, 1)).item(); ta = (run_acc / max(run_n, 1)).item()
            te = time.time()
            ev = evaluate_all(model, kind, sets, dev, bf16=bf16)
            rec = dict(step=int(s), words_seen=int(s * words_per_step), lr=lr_at(min(s, total - 1), total), train_loss=tl, train_acc=ta,
                       elapsed_s=round(time.time() - t0, 1), eval_s=round(time.time() - te, 1), truncated=truncated, eval=ev)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            np.save(os.path.join(adir, "pos", f"step_{s:08d}.npy"), model.pos.weight.detach().float().cpu().numpy().astype(np.float16))
            if s in tok_steps or truncated:
                np.save(os.path.join(adir, "tok", f"step_{s:08d}.npy"), model.tok.weight.detach().float().cpu().numpy().astype(np.float16))
            if s in ckpt_steps or truncated:
                save_original_safetensors(model, os.path.join(adir, "ckpt", f"step_{s:08d}.safetensors"))
            run_loss = torch.zeros((), device=dev); run_acc = torch.zeros((), device=dev); run_n = 0
            done_steps.add(s)
            head = " ".join(f"{k}:{v.get('slot_acc', v.get('planted_slot_acc')):.3f}" for k, v in ev.items())
            log(f"step {s}/{total} train_loss {tl:.4f} train_acc {ta:.4f} | {head} | lr {rec['lr']:.2e}")

        sched_set = set(sched)
        if step == 0:
            snap(0)
        last_resume_save = time.time()
        truncated = False
        model.train()
        status("running")
        while step < total:
            lr = lr_at(step, total)
            for g in opt.param_groups: g["lr"] = lr
            w, p = sampler.sample(B); inp, tgt = mk(w, p)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=bf16):
                lg = model(inp)
            lg = lg.float()
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            step += 1
            run_loss += loss.detach(); run_acc += (lg.argmax(-1) == tgt).float().mean(); run_n += 1
            if step in sched_set:
                snap(step)
            if step % 50 == 0:
                if any(os.path.exists(q) for q in stop_paths) or (a.deadline and time.time() > a.deadline):
                    truncated = True
                    log(f"stopping early at step {step} (stop file or deadline)")
                    break
                if step % 500 == 0:
                    status("running")
                if time.time() - last_resume_save > 1200:
                    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step), resume_path)
                    last_resume_save = time.time()
        if truncated:
            snap(step, truncated=True)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step), resume_path)
            status("truncated")
        else:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step), resume_path)
            status("done")
        log("finished")
        return 0
    except BaseException as e:                         # noqa: report any failure to the orchestrator
        import traceback
        traceback.print_exc()
        try: status("failed", error=repr(e))
        except Exception: pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
