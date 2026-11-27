"""Rough energy and emissions estimate for the study of 25-28 Nov 2026 (see coauthor-note, last section)."""
import json
# tokens from the session's cost record (28 Nov 2026, about 11:00 AEDT); millions
use = {"sonnet": dict(inp=0.127759, cw=3.667361, cr=262.719412, out=1.422890, think=0.720733),
       "opus":   dict(inp=0.069855, cw=2.842490, cr=183.720834, out=0.405849, think=0.141329),
       "haiku":  dict(inp=0.162129, cw=0, cr=0, out=0.014508, think=0.008666)}
# Wh per million tokens.  central: third-party Claude Code rates (input 390, cache write 490, cache read 39, output 1950);
# low: output ~1 Wh per 1000 tokens (Microsoft frontier-model median), others scaled down 4x; high: twice central.
rates = {"low": dict(inp=100, cw=125, cr=10, out=1000), "central": dict(inp=390, cw=490, cr=39, out=1950)}
rates["high"] = {k: 2 * v for k, v in rates["central"].items()}
llm = {}
for case, r in rates.items():
    llm[case] = sum(u["inp"] * r["inp"] + u["cw"] * r["cw"] + u["cr"] * r["cr"] + u["out"] * r["out"] for u in use.values()) / 1000  # kWh
share = {m: sum(u[k] * rates["central"][k] for k in ("inp", "cw", "cr", "out")) / 1000 for m, u in use.items()}
parts = {k: sum(use[m][k] for m in use) * rates["central"][k] / 1000 for k in ("inp", "cw", "cr", "out")}
# Spartan: 4 single-L40S jobs, 5.33 + 5.46 + 8.56 + 9.57 GPU-hours, plus a 2-minute 4-GPU failed start
gpu_h = 5.33 + 5.46 + 8.56 + 9.57 + 4 * 2 / 60
per_gpu_h = {"low": (150 + 50) * 1.2 / 1000, "central": (225 + 75) * 1.35 / 1000, "high": (350 + 100) * 1.5 / 1000}   # (GPU W + host W) * PUE
hpc = {c: gpu_h * v for c, v in per_gpu_h.items()}
sandbox = {"low": 0.1, "central": 0.3, "high": 0.6}   # 2 vCPUs for about 16 h of session time, incl. overhead
ci_dc = {"low": 0.10, "central": 0.35, "high": 0.42}  # kg CO2e/kWh for the model-serving data centres
ci_vic = 0.87                                          # Victoria 2026 coefficient (ESC)
out = {}
for c in ("low", "central", "high"):
    e_llm, e_hpc, e_sb = llm[c], hpc[c], sandbox[c]
    co2 = e_llm * ci_dc[c] + e_hpc * ci_vic + e_sb * ci_dc[c]
    out[c] = dict(kwh_llm=e_llm, kwh_hpc=e_hpc, kwh_sandbox=e_sb, kwh_total=e_llm + e_hpc + e_sb,
                  kg_llm=e_llm * ci_dc[c], kg_hpc=e_hpc * ci_vic, kg_total=co2,
                  water_l_llm=(e_llm + e_sb) * (1.1 if c == "low" else 2.0 if c == "central" else 4.0))
    print(c, {k: round(v, 1) for k, v in out[c].items()})
print("gpu hours", round(gpu_h, 2), "| central kWh by model", {k: round(v, 1) for k, v in share.items()},
      "| central kWh by token type", {k: round(v, 1) for k, v in parts.items()})
json.dump(dict(use_million_tokens=use, rates_wh_per_million=rates, gpu_hours=gpu_h, per_gpu_hour_kwh=per_gpu_h,
               carbon_kg_per_kwh=dict(datacentre=ci_dc, victoria=ci_vic), results=out, central_by_model=share,
               central_by_token_type=parts), open("footprint.json", "w"), indent=1)
