from paths import MODEL_TORN
import time, numpy as np
from np_transformer import NPTransformer
from permdata import *

rng = np.random.default_rng(0)
t0 = time.time()
m = NPTransformer(MODEL_TORN)
print(f"model loaded in {time.time()-t0:.1f}s; vocab={m.E.shape[0]}, D={m.D}, mask matches checkpoint buffer")

# 1. pointer picture == actual permutation (asserted inside)
w = random_words(rng, 2000)
tr = trajectories(w)
print("pointer-chase picture reproduces the permutation on 2000 random words: OK")

# 2. incremental decoder == full forward (check logits on words the model gets right)
w = random_words(rng, 16); p = apply_words(w)
tf_pred, tf_logits = m.teacher_forced_predictions(w, p)
gen, gen_logits = m.generate(w, force_valid=True, return_logits=True)
ok = (gen == p).all(1)
print("free-running exact match on these 16:", ok.sum(), "/16")
if ok.any():
    d = np.abs(gen_logits[ok] - tf_logits[ok]).max()
    print(f"max |logit diff| between cached incremental decoder and full forward: {d:.2e}")

# 3. speed + baseline accuracy
for B in [64]:
    w = random_words(rng, B); p = apply_words(w)
    t0 = time.time(); g = m.generate(w); dt = time.time()-t0
    print(f"generate B={B}: {dt:.2f}s ({B/dt:.1f} words/s); exact-match acc = {(g==p).all(1).mean():.3f}")
