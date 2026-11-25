"""The experiment arms.  Default set (one per GPU of a 4-GPU node): g_m10, g_plant10, g_plant10_b8, e_base."""

PLANT = dict(Lmin=3, Lmax=25)

ARMS = {
    # ---------------------------------------------------------------- default four
    "g_m10": dict(
        kind="general", n_blocks=5,
        desc="CONTROL = the paper's training distribution: natural words over 10 points (relabelled into 25).",
        dist=dict(ms={10: 1.0}, planted_frac=0.0)),
    "g_plant10": dict(
        kind="general", n_blocks=5,
        desc="LONG-CHAIN ENRICHED: 50% natural m=10 words + 50% words with one planted chain of L=3..25 hops through "
             "s=2..10 points, whole word on 10 points (so the word-level support stays 10).",
        dist=dict(ms={10: 1.0}, planted_frac=0.5, plant_m=10, **PLANT)),
    "g_plant10_b8": dict(
        kind="general", n_blocks=8,
        desc="DEPTH: same data as g_plant10 but 8 blocks instead of 5.  If 5 blocks can only chain ~2^4 hops, this one should not have the cliff.",
        dist=dict(ms={10: 1.0}, planted_frac=0.5, plant_m=10, **PLANT)),
    "e_base": dict(
        kind="elementary", n_blocks=5,
        desc="Elementary (adjacent) transpositions, 120 letters, S_16 outputs, partitioned windows: the model of Figure 5b."),
    # ---------------------------------------------------------------- optional extras (ARMS="..." to select)
    "g_plant5": dict(
        kind="general", n_blocks=5,
        desc="SMALLER SUPPORT: same recipe as g_plant10 but every word lives on 5 points (chains have s<=5).",
        dist=dict(ms={5: 1.0}, planted_frac=0.5, plant_m=5, **PLANT)),
    "g_m5": dict(
        kind="general", n_blocks=5,
        desc="Natural words over 5 points only (support-size sweep, no planting).",
        dist=dict(ms={5: 1.0}, planted_frac=0.0)),
    "g_plant10_b3": dict(
        kind="general", n_blocks=3,
        desc="g_plant10 with 3 blocks (does less depth move the cliff down?).",
        dist=dict(ms={10: 1.0}, planted_frac=0.5, plant_m=10, **PLANT)),
}
DEFAULT_ARMS = ["g_m10", "g_plant10", "g_plant10_b8", "e_base"]

# problem-type constants (match the repo configs: torn-bigger-13 and elem-partitioned-long-boosted-1)
SPEC = {
    "general":    dict(vocab=652, n_ctx=50,  input_len=25,  n_out=25),
    "elementary": dict(vocab=34,  n_ctx=136, input_len=120, n_out=16),
}
