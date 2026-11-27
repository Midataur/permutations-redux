"""Check that the committed boosted training files are what the elementary runs trained on: per-slot cross-entropy on
3,000 words of train_data3.csv (commit fdefb8c) against 3,000 validation words. A run that trained on these words should
fit them more tightly, as its wandb training loss is far below its validation loss.  Usage: python elem_train_val_check.py"""
import numpy as np, subprocess, io, pandas as pd, sys
import np_transformer
from np_transformer import NPTransformer
_orig = np_transformer.load_safetensors
np_transformer.load_safetensors = lambda p: ({k.replace('_orig_mod.', ''): v for k, v in _orig(p)[0].items()}, None)
from elem_tools import G, N, apply_words_elem
def git_csv(name, n):
    out=subprocess.run(["git","-C","/home/claude/midataur/permutations","show",f"fdefb8c:scaling-generator/data/elem_partitioned_long_boosted/{name}"],capture_output=True,check=True).stdout
    return pd.read_csv(io.BytesIO(out),header=None,dtype=np.int8,nrows=n).values.astype(np.int64)
tr=git_csv("train_data3.csv",3000); va=git_csv("val_data.csv",3000)
S="/home/claude/snapshots-extracted/model-snapshots/"
MODELS = [("rerun-4",S+"rerun-4/model.safetensors"),("boosted-2",S+"elem-partitioned-long-boosted-2/model.safetensors"),("boosted-1","../scaling-generator/trained-models/elem-partitioned-long-boosted-1/model.safetensors")]
only = sys.argv[1:]
for name,p in MODELS:
    if only and name not in only: continue
    m=NPTransformer(p,max_group=G,input_len=N,num_trans=G)
    out=[]
    for W in (tr,va):
        P=apply_words_elem(W); ce=[]
        for s in range(0,len(W),100):
            _,lg=m.teacher_forced_predictions(W[s:s+100],P[s:s+100])
            lg=lg-lg.max(-1,keepdims=True); lp=lg-np.log(np.exp(lg).sum(-1,keepdims=True))
            ce.append(-np.take_along_axis(lp,(P[s:s+100]+G)[...,None],-1)[...,0])
        ce=np.concatenate(ce); out.append((ce.mean(), np.median(ce), (ce>1e-3).mean()))
    print(f"{name}: training words mean CE {out[0][0]:.2e} (median {out[0][1]:.1e}, share>1e-3 {out[0][2]:.4f}) | validation {out[1][0]:.2e} (median {out[1][1]:.1e}, share>1e-3 {out[1][2]:.4f})", flush=True)
