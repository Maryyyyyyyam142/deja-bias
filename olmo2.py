#!/usr/bin/env python3
"""OLMo 2 extension, as specified in preregistration_olmo2.md (written before any
OLMo 2 likelihood was computed). The same 1B model before and after mid-training on
Dolmino-mix-1124, whose FLAN part carries the leaked SNI bias tasks: three
independent mid-training runs, their merge, then SFT on Tulu 3.

  python3 olmo2.py [likelihoods_csv] [dose_csv] [out_md]
Defaults: results/likelihoods_olmo2.csv data/dose_olmo2.csv results/olmo2.md
"""
import json, sys, numpy as np, pandas as pd, statsmodels.formula.api as smf

LIK, DOSE, OUTP = (sys.argv[1:] + ["results/likelihoods_olmo2.csv", "data/dose_olmo2.csv",
                                    "results/olmo2.md"][len(sys.argv[1:]):])[:3]
L = pd.read_csv(LIK)
V = pd.read_csv("data/variables.csv").merge(pd.read_csv(DOSE), on="sentence_id", how="left")
V[["dose_dolmino", "dose_tulu3"]] = V[["dose_dolmino", "dose_tulu3"]].fillna(0)
ss = json.load(open("data/stereoset_dev.json"))["data"]["intrasentence"]
INST = {f"ss-{s['id']}": i["id"] for i in ss for s in i["sentences"]}
V["item"] = np.where(V.source_dataset == "crows_pairs", V.pair_id, V.sentence_id.map(INST))
STEPS = [("pretraining → mid-training run 1", "olmo2-1b-stage1", "olmo2-1b-ingr1", "dose_dolmino"),
         ("pretraining → mid-training run 2", "olmo2-1b-stage1", "olmo2-1b-ingr2", "dose_dolmino"),
         ("pretraining → mid-training run 3", "olmo2-1b-stage1", "olmo2-1b-ingr3", "dose_dolmino"),
         ("pretraining → released base (merged)", "olmo2-1b-stage1", "olmo2-1b", "dose_dolmino"),
         ("base → SFT", "olmo2-1b", "olmo2-1b-sft", "dose_tulu3")]
STEPS = [s for s in STEPS if {s[1], s[2]} <= set(L.model)]
SCORERS = ("logp_sum", "logp_mean", "full_logp_sum")
EXEMPLARS = ["crows-0-more", "crows-0-less", "crows-50-more", "crows-50-less",
             "ss-62fa295b1bf1e1c85973db251baec1c0"]
OUT = []
def out(s=""): OUT.append(s); print(s)
fp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
W = {sc: L.pivot_table(index="sentence_id", columns="model", values=sc) for sc in SCORERS}

def sentence_frame(sc, before, after, dose):
    S = V.set_index("sentence_id").join((W[sc][after] - W[sc][before]).rename("gain")).reset_index()
    return S.assign(dose=S[dose])

def pair_frame(sc, before, after, dose):
    c = V[V.source_dataset == "crows_pairs"].set_index("sentence_id").join(W[sc][[before, after]])
    m = c[c.pair_role == "stereo"].set_index("pair_id"); l = c[c.pair_role == "antistereo"].set_index("pair_id")
    j = m.join(l, rsuffix="_l")
    return pd.DataFrame(dict(d_before=j[before] - j[before + "_l"], d_after=j[after] - j[after + "_l"],
                             d_dose=np.log1p(j[dose]) - np.log1p(j[dose + "_l"]),
                             d_pile=np.log1p(j.dose_pile) - np.log1p(j.dose_pile_l),
                             d_len=j.sentence_length_words - j.sentence_length_words_l)).assign(
        dd=lambda P: P.d_after - P.d_before)

out("# OLMo 2 extension: the same model before and after it saw the leaked copies")
out()
out("Pre-specified in `preregistration_olmo2.md`. Model: OLMo-2-0425-1B. Exposure:")
n_mid = int((V.dose_dolmino > 0).sum()); n_sft = int((V.dose_tulu3 > 0).sum())
out(f"{n_mid:,} of {len(V):,} benchmark sentences occur in the Dolmino FLAN files "
    f"({int(V.dose_dolmino.sum()):,} copies) and {n_sft:,} in the Tulu 3 SFT mixture "
    f"({int(V.dose_tulu3.sum()):,} copies).")

out("\n## P1. Sentence level: does more exposure mean a larger likelihood gain?\n")
out("`gain ~ log1p(dose) + log1p(dose_pile) + length`, SEs clustered by item.\n")
out("| step | scorer | beta log1p(dose) | 95% CI | p | n |")
out("|---|---|---|---|---|---|")
for lab, b, a, dose in STEPS:
    for sc in SCORERS:
        S = sentence_frame(sc, b, a, dose).dropna(subset=["gain"])
        r = smf.ols("gain ~ np.log1p(dose) + np.log1p(dose_pile) + sentence_length_words", data=S).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(S["item"])[0]})
        k = "np.log1p(dose)"; lo, hi = r.conf_int().loc[k]
        out(f"| {lab} | {sc} | {r.params[k]:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {fp(r.pvalues[k])} | {len(S):,} |")

out("\n## P2. Bias score: does exposure move the within-pair comparison? (CrowS-Pairs)\n")
out("`Δlogp(after) − Δlogp(before) ~ Δlog1p(dose) + Δlog1p(dose_pile) + Δlength`, HC1.")
out("Scores count pairs where the stereotypical sentence is more likely; ties count as not.")
out("The bound removes an exposure effect at either end of the 95% CI from the after-model's")
out("untied pair differences.\n")
out("| step | scorer | beta | 95% CI | p | CrowS before → after | ties after | score at CI ends |")
out("|---|---|---|---|---|---|---|---|")
for lab, b, a, dose in STEPS:
    for sc in SCORERS:
        P = pair_frame(sc, b, a, dose).dropna()
        r = smf.ols("dd ~ d_dose + d_pile + d_len", data=P).fit(cov_type="HC1")
        lo, hi = r.conf_int().loc["d_dose"]; tie = P.d_after == 0
        bound = lambda bb: (((P.d_after - bb * P.d_dose) > 0) & ~tie).mean() * 100
        out(f"| {lab} | {sc} | {r.params.d_dose:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {fp(r.pvalues.d_dose)} | "
            f"{(P.d_before > 0).mean()*100:.2f} → {(P.d_after > 0).mean()*100:.2f} | {int(tie.sum())} | "
            f"{bound(lo):.2f} to {bound(hi):.2f} |")

out("\n## P3. The highest-dose sentences\n")
out("Gain in log-likelihood from before to after, with the sentence's dose for that step.")
out("For the StereoSet sentence, the percentile among all StereoSet gains is in brackets.\n")
out("| step | scorer | " + " | ".join(f"`{e}`" for e in EXEMPLARS) + " |")
out("|---|---|" + "---|" * len(EXEMPLARS))
for lab, b, a, dose in STEPS:
    for sc in ("logp_sum", "full_logp_sum"):
        S = sentence_frame(sc, b, a, dose).set_index("sentence_id")
        pct = (S[S.source_dataset == "stereoset"].gain < S.loc[EXEMPLARS[-1], "gain"]).mean() * 100
        cells = [f"{S.loc[e, 'gain']:+.2f} (dose {int(S.loc[e, 'dose'])})" for e in EXEMPLARS]
        cells[-1] += f" [{pct:.0f}th]"
        out(f"| {lab} | {sc} | " + " | ".join(cells) + " |")

out("\n## Exposed vs unexposed sentences within StereoSet instances (mid-training steps)\n")
out("Instance fixed effects, role, length and Pile dose as controls; CR1 on instance.\n")
out("| step | scorer | beta exposed | 95% CI | p | n sentences / instances |")
out("|---|---|---|---|---|---|")
for lab, b, a, dose in [s for s in STEPS if s[3] == "dose_dolmino"]:
    for sc in ("logp_sum", "logp_mean"):
        T = sentence_frame(sc, b, a, dose).query("source_dataset == 'stereoset'").copy()
        T["exposed"] = (T.dose > 0).astype(float); T["l_pile"] = np.log1p(T.dose_pile)
        T = T[T.groupby("item").exposed.transform("std") > 0]
        for c in ("gain", "exposed", "l_pile", "sentence_length_words"):
            T[c] = T[c] - T.groupby("item")[c].transform("mean")
        r = smf.ols("gain ~ exposed + l_pile + sentence_length_words + C(pair_role) - 1", data=T).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(T["item"])[0]})
        lo, hi = r.conf_int().loc["exposed"]
        out(f"| {lab} | {sc} | {r.params.exposed:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {fp(r.pvalues.exposed)} | "
            f"{len(T):,} / {T['item'].nunique():,} |")

out("\n## Deviations from preregistration_olmo2.md\n")
out("- 20 texts are each shared by two benchmark sentences (duplicate CrowS pairs and StereoSet")
out("  items). Both sentences get the text's count, as with infini-gram. This was decided while")
out("  building `data/dose_olmo2.csv`, before this analysis was first run.")
out("- The SFT checkpoint ships `pytorch_model.bin`, not safetensors, so it was scored after a")
out("  second download. No analysis was changed after scoring.")
out("- This deviations text is the only part of `olmo2.py` edited after the results were seen.")
open(OUTP, "w").write("\n".join(OUT) + "\n")
print(f"\nwrote {OUTP}")
