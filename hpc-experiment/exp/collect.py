"""Bundle the results of a run into one small zip (to upload) and one optional "extras" zip (checkpoints, token embeddings)."""
import glob, json, os, re, time, zipfile
import numpy as np


def _read_jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try: out.append(json.loads(line))
            except Exception: pass
    return out


def _stack(arm_dir, sub):
    files = sorted(glob.glob(os.path.join(arm_dir, sub, "step_*.npy")))
    if not files:
        return None, None
    steps = np.array([int(re.search(r"step_(\d+)", f).group(1)) for f in files])
    return steps, np.stack([np.load(f) for f in files])


def summarize(out_dir, arms):
    """Headline numbers per arm (final snapshot) in one small JSON."""
    summ = {"arms": {}}
    for arm in arms:
        d = os.path.join(out_dir, arm)
        try: st = json.load(open(os.path.join(d, "status.json")))
        except Exception: st = {"state": "unknown"}
        recs = _read_jsonl(os.path.join(d, "metrics.jsonl"))
        info = {"status": st, "n_snapshots": len(recs)}
        if recs:
            last = recs[-1]
            info["final_step"] = last["step"]; info["final_train_loss"] = last["train_loss"]
            ev = last["eval"]
            info["final"] = {k: ({kk: vv for kk, vv in v.items() if kk in ("slot_acc", "word_acc", "loss", "planted_slot_acc")} if isinstance(v, dict) else v)
                             for k, v in ev.items()}
            for k in ("ladder_full", "ladder_in10", "ladder_in5", "cells"):
                if k in ev:
                    info.setdefault("cells", {})[k] = ev[k]["cells"]
        summ["arms"][arm] = info
    return summ


def collect(out_dir, run_id, arms, extra_files=(), make_figs=True):
    t0 = time.time()
    bundle = {}                                   # arcname -> path or bytes
    extras = {}
    root = f"hpc-run-{run_id}"
    # ---- per-arm arrays
    for arm in arms:
        d = os.path.join(out_dir, arm)
        steps, pos = _stack(d, "pos")
        if pos is not None:
            p = os.path.join(out_dir, f"_pos_{arm}.npz"); np.savez_compressed(p, steps=steps, pos=pos)
            bundle[f"{root}/{arm}/pos_emb.npz"] = p
        tsteps, tok = _stack(d, "tok")
        if tok is not None:
            p = os.path.join(out_dir, f"_tok_{arm}.npz"); np.savez_compressed(p, steps=tsteps, tok=tok)
            extras[f"{root}-extras/{arm}/tok_emb.npz"] = p
        for fn in ("metrics.jsonl", "status.json", "config.json", "calib.json"):
            if os.path.exists(os.path.join(d, fn)):
                bundle[f"{root}/{arm}/{fn}"] = os.path.join(d, fn)
        if os.path.exists(os.path.join(d, "train.log")):
            lines = open(os.path.join(d, "train.log"), errors="replace").read().splitlines()
            bundle[f"{root}/{arm}/train.log"] = ("\n".join(lines[:200] + ["... (cut) ..."] + lines[-1500:]) if len(lines) > 1700 else "\n".join(lines)).encode()
        for c in sorted(glob.glob(os.path.join(d, "ckpt", "*.safetensors"))):
            extras[f"{root}-extras/{arm}/ckpt/{os.path.basename(c)}"] = c
    # ---- global files
    for fn in ("env.json", "plan.json", "orchestrator.log"):
        if os.path.exists(os.path.join(out_dir, fn)):
            bundle[f"{root}/{fn}"] = os.path.join(out_dir, fn)
    for p in extra_files:
        if os.path.exists(p):
            bundle[f"{root}/{os.path.basename(p)}"] = p
    if os.path.exists(os.path.join(out_dir, "evalsets.npz")):
        extras[f"{root}-extras/evalsets.npz"] = os.path.join(out_dir, "evalsets.npz")
    summ = summarize(out_dir, arms)
    bundle[f"{root}/summary.json"] = json.dumps(summ, indent=1).encode()
    # ---- figures (best effort)
    if make_figs:
        try:
            from . import figures
            for name, path in figures.make_all(out_dir, arms):
                bundle[f"{root}/figures/{name}"] = path
        except Exception as e:                   # matplotlib missing or a plotting bug must never lose the results
            bundle[f"{root}/figures/FIGURES_FAILED.txt"] = repr(e).encode()
    main_zip = os.path.join(os.path.dirname(out_dir), f"hpc-run-{run_id}.zip")
    ex_zip = os.path.join(os.path.dirname(out_dir), f"hpc-run-{run_id}-extras.zip")
    for zpath, items in ((main_zip, bundle), (ex_zip, extras)):
        if not items:
            continue
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for arc, src in items.items():
                if isinstance(src, bytes):
                    z.writestr(arc, src)
                else:
                    z.write(src, arc)
    return main_zip, (ex_zip if extras else None), time.time() - t0
