"""The only edits to the original config.py: model name (so nothing resumes from old weights or an old wandb run),
number of epochs, and the seed if it is not the original 42.  Each edit must match exactly once.

Usage: python edit_config.py CONFIG MODELNAME EPOCHS SEED
"""
import sys

path, name, epochs, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
s = open(path).read()
edits = [('MODELNAME = "torn-bigger-13"', f'MODELNAME = "{name}"'),
         ("num_epochs = 10**8", f"num_epochs = {epochs}")]
if seed != 42:
    edits.append(("set_seed(42)", f"set_seed({seed})"))
for old, new in edits:
    assert s.count(old) == 1, f"expected exactly one '{old}' in {path}"
    s = s.replace(old, new)
for must in ('DATA = "/data/torn_bigger/"', "BATCHSIZE = 1024", "n_blocks = 5", "learning_rate = 3*(10**-4)",
             "weight_decay = 0.05", "lr_patience = 10", 'INPUT_TYPE = "general"', "RELABEL = True"):
    assert must in s, f"config does not look like the torn-bigger-13 config: missing '{must}'"
open(path, "w").write(s)
print("config: " + "; ".join(f"{o} -> {n}" for o, n in edits))
