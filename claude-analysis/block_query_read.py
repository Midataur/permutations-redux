"""
Do the query positions read the word just after the block boundary?  For each model and each head, the attention from
the 16 query positions to the word positions (mean over N test words); the share of that attention on positions
[b, b+12) compared with the share on 12-position windows elsewhere (b = the model's boundary; 72 for models without one).
Usage: python block_query_read.py TEST_CSV N name=path:blocks:b ...  -> block_query_read.json
"""
import sys, json
import numpy as np
import np_transformer
from np_transformer import NPTransformer
from elem_tools import G, N as NW, apply_words_elem
_orig = np_transformer.load_safetensors
np_transformer.load_safetensors = lambda p: ({k.replace("_orig_mod.", ""): v for k, v in _orig(p)[0].items()}, None)
words = np.loadtxt(sys.argv[1], delimiter=",", dtype=np.int64)[:int(sys.argv[2])]
perms = apply_words_elem(words)
res = {}
for name, rest in (a.split("=", 1) for a in sys.argv[3:]):
    path, blocks, b = rest.rsplit(":", 2); b = int(b)
    m = NPTransformer(path, max_group=G, input_len=NW, num_trans=G, n_blocks=int(blocks))
    toks = np.concatenate([words, np.full((len(words), 1), m.START), perms[:, :-1] + m.num_trans], 1)
    A = 0
    for s in range(0, len(words), 50):
        _, at = m.forward(toks[s:s + 50], return_attn=True)
        A = A + np.stack([x[:, :, NW:, :NW].sum(0).mean(1) for x in at])     # (layers, heads, 120)
    A /= len(words)
    A = A / A.sum(-1, keepdims=True)                                       # share of word-position attention
    win = lambda c: A[..., c:c + 12].sum(-1)
    starts = range(0, NW - 12 + 1)
    W = np.stack([win(c) for c in starts], -1)                             # (layers, heads, windows)
    at_b = win(b)
    rank = (W > at_b[..., None]).sum(-1)                                   # how many windows get more than [b, b+12)
    best = []
    for li in range(A.shape[0]):
        for h in range(A.shape[1]):
            c = int(np.argmax(W[li, h]))
            best.append(dict(block=li + 1, head=h, share_at_b=round(float(at_b[li, h]), 3), uniform=0.1,
                             top_window_start=c, top_share=round(float(W[li, h, c]), 3), rank_of_b=int(rank[li, h])))
    hits = [x for x in best if abs(x["top_window_start"] - b) <= 3 and x["top_share"] > 0.2]
    res[name] = dict(boundary=b, heads_whose_top_window_starts_at_b=hits, all_heads=best)
    print(f"{name:26s} b={b:3d}: heads whose most-read 12-letter window starts within 3 of b (share > 0.2):",
          [(x['block'], x['head'], x['top_window_start'], x['top_share']) for x in hits], flush=True)
json.dump(res, open("block_query_read.json", "w"), indent=1)
