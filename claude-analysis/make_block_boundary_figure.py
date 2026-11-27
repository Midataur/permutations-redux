import json, numpy as np, sys
sys.path.insert(0, "/home/claude/permutations-redux/claude-analysis")
from np_transformer import load_safetensors
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=sys.argv[1].rstrip("/")+"/" if len(sys.argv)>1 else "/home/claude/snapshots-extracted/model-snapshots/"
names=["elem-partitioned-long-boosted-1","elem-partitioned-long-boosted-2","elem-partitioned-long-boosted-4","rerun-1","rerun-2","rerun-3","rerun-4","elem-partitioned-long-11","elem-partitioned-long-12","elem-partitioned-long-14"]
res=json.load(open("block_survey.json"))  # run from claude-analysis/ after block_survey.py
fig,axs=plt.subplots(2,len(names),figsize=(3.0*len(names),6.2),gridspec_kw=dict(height_ratios=[3,1]))
for i,n in enumerate(names):
    T,_=load_safetensors(R+n+"/model.safetensors"); P=[T[k] for k in T if k.endswith("position_embedding.weight")][0].astype(float)
    Q=P/np.linalg.norm(P,axis=1,keepdims=True); C=Q@Q.T
    ax=axs[0,i]; ax.imshow(C,cmap="RdBu_r",vmin=-.6,vmax=.6); b=res[n]["boundary_top"][0][0]
    ax.axhline(b-.5,color="g",lw=.6); ax.axvline(b-.5,color="g",lw=.6); ax.axhline(119.5,color="k",lw=.4); ax.axvline(119.5,color="k",lw=.4)
    ax.set_title(n.replace("elem-partitioned-long","epl"),fontsize=8); ax.set_xticks([0,72,120]); ax.set_yticks([0,72,120]); ax.tick_params(labelsize=6)
    # boundary profile
    nr=lambda v:(v-v.mean())/(np.linalg.norm(v-v.mean())+1e-9); ks=range(5,114); d=[]
    for k in ks:
        J=np.array([j for j in range(C.shape[0]) if abs(j-k)>6 and abs(j-k-1)>6]); d.append(1-nr(C[k,J])@nr(C[k+1,J]))
    d=np.array(d); z=(d-np.median(d))/(1.4826*np.median(np.abs(d-np.median(d))))
    axs[1,i].plot([k+1 for k in ks],z,lw=.8); axs[1,i].axvline(72,color="k",ls=":",lw=.6); axs[1,i].set_ylim(-3,max(8,z.max()+1)); axs[1,i].tick_params(labelsize=6)
    axs[1,i].set_xlabel("boundary b (z of row change)",fontsize=6)
plt.tight_layout(); plt.savefig("fig_block_boundaries.png",dpi=120)
