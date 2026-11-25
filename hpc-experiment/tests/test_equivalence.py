"""exp.model.Net  ==  the repo's scaling-generator/transformer.py  (same logits, same checkpoint layout).
Usage: python tests/test_equivalence.py general|elementary      (needs torch; accelerate/sklearn are stubbed)"""
import importlib, os, sys, tempfile, types
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SG = os.path.join(REPO, "scaling-generator")
kind = sys.argv[1] if len(sys.argv) > 1 else "general"

# ---- stubs so that the repo's own modules import without accelerate / sklearn
def stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
class _Acc:
    def __init__(self, *a, **k): pass
stub("accelerate", Accelerator=_Acc, load_checkpoint_and_dispatch=lambda *a, **k: None)
stub("accelerate.utils", set_seed=lambda s: None)
class _B: pass
class _T: pass
class _S: pass
stub("sklearn"); stub("sklearn.base", BaseEstimator=_B, TransformerMixin=_T); stub("sklearn.decomposition", TruncatedSVD=_S)
tmp = tempfile.mkdtemp()
cfg_src = (os.path.join(SG, "trained-models", "torn-bigger-13-config.py") if kind == "general" else os.path.join(SG, "config.py"))
open(os.path.join(tmp, "config.py"), "w").write(open(cfg_src).read())
sys.path.insert(0, tmp); sys.path.insert(1, SG)
import config as C                                    # the repo's own config for this model
import transformer as R                               # the repo's own transformer.py
sys.path.insert(0, os.path.join(HERE, ".."))
from exp.model import Net

print(kind, "config:", C.MODELNAME, "vocab", C.vocab_size, "ctx", C.CONTEXT_LENGTH, "input", C.INPUT_LENGTH, "out", C.MAX_GROUP_SIZE, "type", C.INPUT_TYPE)
torch.manual_seed(0)
ref = R.Transformer().eval()
mine = Net(C.vocab_size, C.CONTEXT_LENGTH, C.INPUT_LENGTH, C.MAX_GROUP_SIZE, D=C.n_embed, H=C.n_head, n_blocks=C.n_blocks).eval()
mine.load_original_state_dict(ref.state_dict())
idx = torch.randint(0, C.vocab_size - 2, (6, C.CONTEXT_LENGTH))
with torch.no_grad():
    a, b = ref(idx), mine(idx)
err = (a - b).abs().max().item()
print("logits:", tuple(a.shape), "max abs diff", err, "ref scale", a.abs().max().item())
assert a.shape == b.shape and err < 5e-4 * max(1, a.abs().max().item()), "MODEL MISMATCH"
sd_ref, sd_mine = ref.state_dict(), mine.original_state_dict()
assert set(sd_ref) == set(sd_mine), (set(sd_ref) ^ set(sd_mine))
worst = max((sd_ref[k].float() - sd_mine[k].float()).abs().max().item() for k in sd_ref)
print("state dict: same", len(sd_ref), "keys; worst difference after the round trip:", worst)
assert worst == 0.0
# initialisation statistics match (fresh models, different seeds, same shapes and families)
fresh = Net(C.vocab_size, C.CONTEXT_LENGTH, C.INPUT_LENGTH, C.MAX_GROUP_SIZE, D=C.n_embed, H=C.n_head, n_blocks=C.n_blocks)
for k in ("blocks.0.sa.heads.0.query.weight", "blocks.0.ffwd.net.0.weight", "token_embedding_table.weight", "lm_head.weight"):
    s1, s2 = sd_ref[k].std().item(), fresh.original_state_dict()[k].std().item()
    print(f"  init std {k}: repo {s1:.4f} mine {s2:.4f}")
    assert abs(s1 - s2) / s1 < 0.05
# the safetensors file written by exp.model is readable by the claude-analysis NumPy port and gives the same logits
from exp.model import save_original_safetensors
path = os.path.join(tmp, "m.safetensors"); save_original_safetensors(mine, path)
sys.path.insert(0, os.path.join(REPO, "claude-analysis"))
from np_transformer import NPTransformer
npm = NPTransformer(path, C.MAX_GROUP_SIZE, C.INPUT_LENGTH, num_trans=C.num_trans)
with torch.no_grad():
    lg32 = Net(C.vocab_size, C.CONTEXT_LENGTH, C.INPUT_LENGTH, C.MAX_GROUP_SIZE, D=C.n_embed, H=C.n_head, n_blocks=C.n_blocks)
    # reload the bfloat16 file into the torch model to compare like with like
    from safetensors.torch import load_file
    lg32.load_original_state_dict({k: v.float() for k, v in load_file(path).items()}); lg32.eval()
    t_logits = lg32(idx).numpy()
n_logits = npm.forward(idx.numpy())
err2 = np.abs(t_logits - n_logits).max()
print("NumPy port vs torch on the saved bf16 file: max abs diff", err2, "scale", np.abs(t_logits).max())
assert err2 < 2e-3 * max(1, np.abs(t_logits).max())
print("OK", kind)
