"""Diagnostic figures (matplotlib, best effort).  Colours: Okabe-Ito (colour-blind safe); every line is also labelled."""
import glob, json, os, re
import numpy as np

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _recs(out_dir, arm):
    p = os.path.join(out_dir, arm, "metrics.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def make_all(out_dir, arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(out_dir, "_figures"); os.makedirs(figdir, exist_ok=True)
    out = []
    col = {a: OKABE[i % len(OKABE)] for i, a in enumerate(arms)}
    garms = [a for a in arms if a.startswith("g_")]
    # ---------------- 1. learning curves (general arms)
    if garms:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, (setname, ttl) in zip(axs, (("nat25", "natural S25 words (paper-style test)"), ("nat10", "natural 10-point words"), ("ladder_full", "planted chains, planted slot only"))):
            for a in garms:
                r = _recs(out_dir, a)
                if not r: continue
                x = [q["step"] for q in r if setname in q["eval"]]
                y = [q["eval"][setname].get("slot_acc", q["eval"][setname].get("planted_slot_acc")) for q in r if setname in q["eval"]]
                ax.plot(np.maximum(x, 1), y, color=col[a], lw=1.6, label=a)
            ax.set_xscale("log"); ax.set_title(ttl, fontsize=10); ax.set_xlabel("optimiser step"); ax.grid(alpha=.25)
        axs[0].set_ylabel("slot accuracy"); axs[0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); p = os.path.join(figdir, "learning_curves_general.png"); fig.savefig(p, dpi=130); plt.close(fig); out.append(("learning_curves_general.png", p))
        # ---------------- 2. the hop cliff at the end of training
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, (setname, s, ttl) in zip(axs, (("ladder_full", 6, "out of support (25-point words), s=6"), ("ladder_full", 14, "out of support, s=14"), ("ladder_in10", 6, "inside the 10-point support, s=6"))):
            for a in garms:
                r = _recs(out_dir, a)
                if not r: continue
                cells = r[-1]["eval"].get(setname, {}).get("cells", {})
                pts = sorted((int(k[1:].split("_")[0]), v[0]) for k, v in cells.items() if k.endswith(f"_s{s}"))
                if pts: ax.plot(*zip(*pts), "-o", ms=4, color=col[a], lw=1.6, label=a)
            ax.set_title(ttl, fontsize=10); ax.set_xlabel("hops L in the planted chain"); ax.grid(alpha=.25); ax.set_ylim(0, 1.02)
        axs[0].set_ylabel("accuracy at the planted slot"); axs[0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); p = os.path.join(figdir, "hop_cliff_final.png"); fig.savefig(p, dpi=130); plt.close(fig); out.append(("hop_cliff_final.png", p))
    # ---------------- 3. embedding pictures during training
    for a in arms:
        files = sorted(glob.glob(os.path.join(out_dir, a, "pos", "step_*.npy")))
        if len(files) < 2: continue
        idx = np.unique(np.linspace(0, len(files) - 1, 8).round().astype(int))
        fig, axs = plt.subplots(2, 4, figsize=(16, 8.4))
        for ax, i in zip(axs.ravel(), idx):
            P = np.load(files[i]).astype(np.float32); P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
            im = ax.imshow(P @ P.T, vmin=-0.2, vmax=1.0, cmap="viridis")
            ax.set_title("step " + str(int(re.search(r"step_(\d+)", files[i]).group(1))), fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        for ax in axs.ravel()[len(idx):]: ax.axis("off")
        fig.suptitle(f"{a}: cosine similarity of the position embeddings during training", fontsize=11)
        fig.colorbar(im, ax=axs, shrink=0.6, label="cosine")
        p = os.path.join(figdir, f"pos_pictures_{a}.png"); fig.savefig(p, dpi=110); plt.close(fig); out.append((f"pos_pictures_{a}.png", p))
    return out
