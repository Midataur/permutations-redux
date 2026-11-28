"""
D-RASP lines of the program a stage-1 graph decompiles to, counted as in convert_to_code.py of Huang et al.:
one line per selector (query-variable x key-variable product), one per aggregate (one per value-path variable), one per
MLP (unsplit), one per projection to the vocabulary, plus a bias line and the prediction line.  Variables are paths:
token and pos; each head creates one variable per variable feeding its value input; each MLP creates one variable.
Keeping every query-key product gives an upper bound on the paper's stage-3 count; for the full graph it reproduces their
determine_total_line (165,818,738 for 5 layers and 6 heads, unsplit MLPs).
"""
def count_lines(kept, L=5, H=6, split_mlps=False):
    """kept: iterable of (dest, source) with dest = (l,'q'|'k'|'v',h) | (l,'mlp') | ('lm',)."""
    ins = {}
    for d, s in kept:
        ins.setdefault(tuple(d), set()).add(s)
    nvars = {"tok": 1, "pos": 1}
    lines = 0
    for l in range(L):
        for h in range(H):
            q, k, v = (ins.get((l, a, h), set()) for a in ("q", "k", "v"))
            if not v:
                nvars[f"a{l}.{h}"] = 0; continue
            nq = sum(nvars.get(s, 0) for s in q); nk = sum(nvars.get(s, 0) for s in k); nv = sum(nvars.get(s, 0) for s in v)
            lines += nq * nk + nv
            nvars[f"a{l}.{h}"] = nv
        mi = ins.get((l, "mlp"), set())
        nin = sum(nvars.get(s, 0) for s in mi)
        if split_mlps:                       # one MLP application, and one output variable, per input variable
            lines += nin; nvars[f"m{l}"] = nin
        elif nin > 0:
            lines += 1; nvars[f"m{l}"] = 1
        else:
            nvars[f"m{l}"] = 0
    lines += sum(nvars.get(s, 0) for s in ins.get(("lm",), set())) + 2
    return lines


if __name__ == "__main__":
    L, H = 5, 6
    SRC = ["tok", "pos"]
    for l in range(L):
        SRC += [f"a{l}.{h}" for h in range(H)] + [f"m{l}"]
    nb = lambda l, mlp=False: 2 + l * (H + 1) + (H if mlp else 0)
    full = [((l, a, h), s) for l in range(L) for a in ("q", "k", "v") for h in range(H) for s in SRC[:nb(l)]]
    full += [((l, "mlp"), s) for l in range(L) for s in SRC[:nb(l, True)]] + [(("lm",), s) for s in SRC]
    print("full graph:", count_lines(full), "(determine_total_line: 165818738)")
    print("full graph, split MLPs:", count_lines(full, split_mlps=True), "(determine_total_line: 35602724216)")
