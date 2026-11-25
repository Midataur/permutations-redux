import numpy as np
from permdata import apply_words


def tf_eval(model, words, bs=100, n=25):
    """Teacher-forced evaluation. Returns perms, correct (B,n) bool, margin (B,n) = true-token logit
    minus best competing logit, pred (B,n)."""
    perms = apply_words(words, n)
    B = len(words)
    correct = np.zeros((B, n), bool); margin = np.zeros((B, n), np.float32); pred = np.zeros((B, n), np.int16)
    for s in range(0, B, bs):
        p, lg = model.teacher_forced_predictions(words[s:s + bs], perms[s:s + bs])
        tgt = perms[s:s + bs] + model.num_trans
        true_l = np.take_along_axis(lg, tgt[..., None], -1)[..., 0]
        lg2 = lg.copy()
        np.put_along_axis(lg2, tgt[..., None], -np.inf, -1)
        margin[s:s + bs] = true_l - lg2.max(-1)
        pred[s:s + bs] = p
        correct[s:s + bs] = p == perms[s:s + bs]
    return perms, correct, margin, pred
