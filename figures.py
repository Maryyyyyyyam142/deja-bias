#!/usr/bin/env python3
"""Phase A figures. Palette: dataviz reference categorical slots 1-3, fixed order,
colour following the model (entity), never its rank."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # blue, orange, aqua
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a86", "#e6e6e3"
COL = {"olmo-7b-v1.7": S1, "olmo-7b-v1.5": S2, "pythia-6.9b": S3}
LBL = {"olmo-7b-v1.7": "OLMo-7B v1.7  (contaminated)",
       "olmo-7b-v1.5": "OLMo-7B v1.5  (clean)",
       "pythia-6.9b":  "Pythia-6.9B  (clean)"}
ORDER = ["olmo-7b-v1.7", "olmo-7b-v1.5", "pythia-6.9b"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.spines.top": False, "axes.spines.right": False})

L = pd.read_csv("results/likelihoods.csv"); V = pd.read_csv("data/variables.csv")
D = L.merge(V, on="sentence_id"); D["l_dolma"] = np.log1p(D.dose_dolma); D["l_pile"] = np.log1p(D.dose_pile)

def pair_frame(scorer="logp_sum"):
    c = D[D.source_dataset == "crows_pairs"].copy(); c["y"] = c[scorer]
    m = c[c.pair_role == "stereo"].set_index(["pair_id", "model"])
    l = c[c.pair_role == "antistereo"].set_index(["pair_id", "model"])
    j = m.join(l, rsuffix="_l", how="inner")
    return pd.DataFrame(dict(d_logp=j.y - j.y_l, d_dose=j.l_dolma - j.l_dolma_l,
                             d_pile=j.l_pile - j.l_pile_l,
                             d_len=j.sentence_length_words - j.sentence_length_words_l,
                             bias_type=j.bias_type)).reset_index()

# ---- fig 1: dose-response, small multiples -------------------------------
P = pair_frame()
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True, sharey=True)
for ax, mdl in zip(axes, ORDER):
    s = P[P.model == mdl].dropna(subset=["d_logp", "d_dose"])
    ax.axhline(0, color=GRID, lw=1, zorder=0); ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.scatter(s.d_dose, s.d_logp, s=7, alpha=.28, color=COL[mdl], linewidths=0, zorder=2)
    g, _ = pd.factorize(s.bias_type)
    r = smf.ols("d_logp ~ d_dose", data=s).fit(cov_type="cluster", cov_kwds={"groups": g})
    xs = np.linspace(s.d_dose.min(), s.d_dose.max(), 60)
    pr = r.get_prediction(pd.DataFrame({"d_dose": xs})).summary_frame(alpha=.05)
    ax.fill_between(xs, pr.mean_ci_lower, pr.mean_ci_upper, color=COL[mdl], alpha=.18, lw=0, zorder=3)
    ax.plot(xs, pr["mean"], color=COL[mdl], lw=2, zorder=4)
    b, p = r.params["d_dose"], r.pvalues["d_dose"]
    ax.set_title(LBL[mdl], fontsize=9, loc="left")
    ax.text(.03, .04, f"β = {b:+.3f}   p = {p:.2f}", transform=ax.transAxes,
            fontsize=8.5, color=INK2)
    ax.set_xlabel("Δ log(1+dose_dolma)   (stereo − antistereo)")
axes[0].set_ylabel("Δ log-likelihood (nats)")
fig.suptitle("No dose–response within CrowS-Pairs minimal pairs, in any arm",
             fontsize=11, x=.005, ha="left", color=INK)
fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig("figures/fig1_dose_response.png", dpi=200)
print("fig1 written")

# ---- fig 2: the placebo result ------------------------------------------
rows = []
for scorer, lab in (("logp_sum", "sum log-prob"), ("logp_mean", "mean log-prob")):
    Q = pair_frame(scorer)
    for mdl in ORDER:
        s = Q[Q.model == mdl].dropna(subset=["d_logp", "d_dose", "d_pile", "d_len"])
        thr = s.d_dose.abs().quantile(.9); t = s[s.d_dose.abs() >= thr]
        g, _ = pd.factorize(t.bias_type)
        r = smf.ols("d_logp ~ d_dose + d_pile + d_len", data=t).fit(cov_type="cluster", cov_kwds={"groups": g})
        ci = r.conf_int().loc["d_dose"]
        rows.append((lab, mdl, r.params["d_dose"], ci[0], ci[1], r.pvalues["d_dose"]))
R = pd.DataFrame(rows, columns=["scorer", "model", "b", "lo", "hi", "p"])
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), sharey=True)
for ax, lab in zip(axes, R.scorer.unique()):
    sub = R[R.scorer == lab]
    for i, mdl in enumerate(ORDER):
        r = sub[sub.model == mdl].iloc[0]; y = len(ORDER) - 1 - i
        ax.plot([r.lo, r.hi], [y, y], color=COL[mdl], lw=2.5, solid_capstyle="round", zorder=3)
        ax.scatter([r.b], [y], s=60, color=COL[mdl], zorder=4,
                   edgecolors="#fcfcfb", linewidths=2)
        ax.text(r.hi + .03, y, f"p = {r.p:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([LBL[m] for m in reversed(ORDER)], fontsize=8.5)
    ax.set_title(f"scorer: {lab}", fontsize=9, loc="left"); ax.set_xlabel("β on Δ log-dose, top-decile |Δdose|")
fig.suptitle("The tail effect appears in the CLEAN arms too — so it is not contamination",
             fontsize=11, x=.005, ha="left", color=INK)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig("figures/fig2_placebo.png", dpi=200)
print("fig2 written")

# ---- fig 3: decontaminated StereoSet score ------------------------------
import json
ss = json.load(open("data/stereoset_dev.json"))["data"]["intrasentence"]
role = {"stereotype": "stereo", "anti-stereotype": "antistereo", "unrelated": "unrelated"}
inst = [{role[s["gold_label"]]: f"ss-{s['id']}" for s in i["sentences"] if s["gold_label"] in role} for i in ss]
inst = [d for d in inst if len(d) == 3]
V2 = V.set_index("sentence_id")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
for ax, scorer in zip(axes, ("logp_sum", "logp_mean")):
    for i, mdl in enumerate(ORDER):
        y = D[D.model == mdl].set_index("sentence_id")[scorer]
        a = np.array([[y[d["stereo"]], y[d["antistereo"]], y[d["unrelated"]],
                       sum(V2.loc[d[k], "dose_natinst"] for k in d)] for d in inst
                      if all(k in y.index for k in d.values())])
        full = (a[:, 0] > a[:, 1]).mean() * 100
        mz = a[:, 3] == 0; idx = np.where(mz)[0]
        rng = np.random.default_rng(0)
        bs = [ (a[r, 0] > a[r, 1]).mean()*100 for r in (rng.choice(idx, len(idx)) for _ in range(2000)) ]
        lo, hi = np.percentile(bs, [2.5, 97.5]); zero = (a[mz, 0] > a[mz, 1]).mean()*100
        yy = len(ORDER) - 1 - i
        ax.plot([lo, hi], [yy, yy], color=COL[mdl], lw=2.5, solid_capstyle="round", zorder=3)
        ax.scatter([zero], [yy], s=60, color=COL[mdl], zorder=4, edgecolors="#fcfcfb", linewidths=2)
        ax.scatter([full], [yy], s=70, facecolors="none", edgecolors=COL[mdl], linewidths=2, zorder=4)
        ax.text(full + .6, yy + .17, "full", fontsize=8, color=INK2)
    ax.axvline(50, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([LBL[m] for m in reversed(ORDER)], fontsize=8.5)
    ax.set_title(f"scorer: {scorer}", fontsize=9, loc="left")
    ax.set_xlabel("StereoSet stereotype score  (filled = zero-dose subset, 95% bootstrap CI)")
fig.suptitle("Decontaminated StereoSet score: gaps are within noise and flip sign across scorers",
             fontsize=11, x=.005, ha="left", color=INK)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig("figures/fig3_decontaminated.png", dpi=200)
print("fig3 written")
