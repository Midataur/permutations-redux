"""Per-snapshot evaluation: teacher-forced accuracy / loss on the fixed evaluation sets, split by hop count and by
number of distinct points (general) or width (elementary); planted-chain cells are scored on the planted slot only."""
import numpy as np
import torch
import torch.nn.functional as F
from .datagen import general_batch, elem_batch


@torch.no_grad()
def run_set(model, kind, words, perm, device, bs=1000, bf16=True):
    """Returns correct (B, n_out) bool numpy and mean cross-entropy per slot (B, n_out)."""
    model.eval()
    mk = general_batch if kind == "general" else elem_batch
    corr, loss = [], []
    for i in range(0, len(words), bs):
        w = torch.as_tensor(words[i:i + bs], dtype=torch.long, device=device)
        p = torch.as_tensor(perm[i:i + bs], dtype=torch.long, device=device)
        inp, tgt = mk(w, p)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=bf16):
            lg = model(inp)
        lg = lg.float()
        corr.append((lg.argmax(-1) == tgt).cpu().numpy())
        loss.append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape).cpu().numpy())
    model.train()
    return np.concatenate(corr), np.concatenate(loss)


def _by(values, correct, vmax):
    """accuracy and count for each value 0..vmax of an integer array (same shape as correct)."""
    v = np.clip(values.ravel(), 0, vmax)
    c = correct.ravel().astype(np.float64)
    cnt = np.bincount(v, minlength=vmax + 1)
    acc = np.bincount(v, weights=c, minlength=vmax + 1) / np.maximum(cnt, 1)
    return [None if n == 0 else round(float(a), 5) for a, n in zip(acc, cnt)], cnt.tolist()


def evaluate_all(model, kind, sets, device, bf16=True):
    res = {}
    names = sorted({k.split("__")[0] for k in sets})
    for name in names:
        words, perm = sets[f"{name}__words"], sets[f"{name}__perm"]
        corr, loss = run_set(model, kind, words, perm, device, bf16=bf16)
        r = {}
        if f"{name}__k" in sets:                                       # planted cells: planted slot only
            ks = sets[f"{name}__k"]; idx = np.arange(len(ks))
            c = corr[idx, ks]
            ax2 = "s" if f"{name}__s" in sets else "w"
            A, B_ = sets[f"{name}__L"], sets[f"{name}__{ax2}"]
            cells = {}
            for a in np.unique(A):
                for b in np.unique(B_):
                    m = (A == a) & (B_ == b)
                    if m.any():
                        cells[f"L{a}_{ax2}{b}"] = [round(float(c[m].mean()), 4), int(m.sum())]
            r["cells"] = cells
            r["planted_slot_acc"] = round(float(c.mean()), 4)
        else:
            r["slot_acc"] = round(float(corr.mean()), 5)
            r["word_acc"] = round(float(corr.all(1).mean()), 4)
            r["loss"] = round(float(loss.mean()), 6)
            hmax = 25 if kind == "general" else 100
            r["acc_by_hops"], r["n_by_hops"] = _by(sets[f"{name}__hops"], corr, hmax)
            key2, vmax2 = ("distinct", 25) if kind == "general" else ("width", 16)
            r[f"acc_by_{key2}"], r[f"n_by_{key2}"] = _by(sets[f"{name}__{key2}"], corr, vmax2)
        res[name] = r
    return res
