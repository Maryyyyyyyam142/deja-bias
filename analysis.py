#!/usr/bin/env python3
"""Task 2 section 5 analyses. Pre-registered in preregistration.md."""
import numpy as np, pandas as pd, statsmodels.formula.api as smf, json, sys

L = pd.read_csv("results/likelihoods.csv")
V = pd.read_csv("data/variables.csv")
D = L.merge(V, on="sentence_id", how="inner")
D["l_dolma"] = np.log1p(D["dose_dolma"]); D["l_pile"] = np.log1p(D["dose_pile"])
D["l_natinst"] = np.log1p(D["dose_natinst"])

def pairs(scorer):
    """CrowS pair-level frame: one row per (pair, model)."""
    c = D[D.source_dataset == "crows_pairs"].copy()
    c["y"] = c[scorer]
    m = c[c.pair_role == "stereo"].set_index(["pair_id", "model"])
    l = c[c.pair_role == "antistereo"].set_index(["pair_id", "model"])
    j = m.join(l, rsuffix="_l", how="inner")
    j = j[j.l_dolma.notna() & j.l_dolma_l.notna()]
    out = pd.DataFrame(dict(
        d_logp = j.y - j.y_l,
        d_dose = j.l_dolma - j.l_dolma_l,
        d_pile = (j.l_pile - j.l_pile_l),
        d_len  = j.sentence_length_words - j.sentence_length_words_l,
        bias_type = j.bias_type, arm = j.arm, corpus = j.corpus,
    )).reset_index()
    return out

def fit(df, formula, label):
    """OLS with SEs clustered by bias_type. Complete cases are selected BEFORE
    fitting so the cluster vector stays aligned with the estimation sample."""
    cols = [c for c in ("d_logp", "d_dose", "d_pile", "d_len") if c in formula]
    d = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    if "d_pile" in formula and (df.d_pile.notna().mean() < 0.5):
        formula = formula.replace(" + d_pile", "")
        label += " [NO pile covariate - INCOMPLETE]"
        d = df.dropna(subset=["d_logp", "d_dose", "d_len"]).copy()
    if len(d) < 30 or d.d_dose.std() == 0:
        print(f"  {label:44s} SKIPPED (n={len(d)}, sd(d_dose)={d.d_dose.std():.4f})"); return None
    g, _ = pd.factorize(d.bias_type)
    r = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": g})
    df = d
    b, se, p = r.params.get("d_dose", np.nan), r.bse.get("d_dose", np.nan), r.pvalues.get("d_dose", np.nan)
    ci = r.conf_int().loc["d_dose"] if "d_dose" in r.params else [np.nan, np.nan]
    print(f"  {label:44s} beta={b:+.4f}  se={se:.4f}  95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  p={p:.4f}  n={len(df)}")
    return r

def main():
    print("="*96); print("DATA"); print("="*96)
    print(f"  likelihood rows {len(L)}, models {sorted(L.model.unique())}")
    have = D.groupby("model").apply(lambda g: g.dose_dolma.notna().sum(), include_groups=False)
    print(f"  rows with dose_dolma per model:\n{have.to_string()}")

    for scorer in ("logp_sum", "logp_mean"):
        print("\n" + "="*96)
        print(f"5.1 WITHIN-PAIR (CrowS)   scorer = {scorer}")
        print("="*96)
        P = pairs(scorer)
        if P.empty: print("  no pair rows yet"); continue
        F = "d_logp ~ d_dose + d_pile + d_len"
        for mdl in sorted(P.model.unique()):
            s = P[P.model == mdl]
            arm = s.arm.iloc[0]
            fit(s, F, f"{mdl} [{arm}]")

        # H3 difference-in-differences, OLMo pair (same tokenizer/arch/lab)
        olmo = P[P.model.str.startswith("olmo")].copy()
        if olmo.model.nunique() == 2:
            olmo["contam"] = (olmo.arm == "contaminated").astype(int)
            f2 = "d_logp ~ d_dose * contam + d_pile + d_len + C(model)"
            if olmo.d_pile.notna().mean() < 0.5: f2 = f2.replace(" + d_pile", "")
            olmo = olmo.dropna(subset=[c for c in ("d_logp","d_dose","d_pile","d_len") if c in f2])
            g2, _ = pd.factorize(olmo.bias_type)
            r = smf.ols(f2, data=olmo).fit(cov_type="cluster", cov_kwds={"groups": g2})
            k = "d_dose:contam"
            if k in r.params:
                ci = r.conf_int().loc[k]
                print(f"\n  H3 beta_DiD (OLMo v1.7 - v1.5)            "
                      f"beta={r.params[k]:+.4f}  se={r.bse[k]:.4f}  "
                      f"95%CI[{ci[0]:+.4f},{ci[1]:+.4f}]  p={r.pvalues[k]:.4f}")

        # H4 sign agreement
        print("\n  H4 sign agreement P(sign d_dose == sign d_logp):")
        for mdl in sorted(P.model.unique()):
            s = P[(P.model == mdl) & (P.d_dose != 0)]
            if len(s):
                agree = (np.sign(s.d_dose) == np.sign(s.d_logp)).mean()
                se = np.sqrt(agree*(1-agree)/len(s))
                print(f"    {mdl:22s} {agree*100:5.1f}%  +/-{1.96*se*100:.1f}  n={len(s)}")

        # H5 tail-driven
        print("\n  H5 tail: beta among top-decile |d_dose| vs all")
        for mdl in sorted(P.model.unique()):
            s = P[P.model == mdl]
            if len(s) < 100: continue
            thr = s.d_dose.abs().quantile(0.9)
            fit(s[s.d_dose.abs() >= thr], F, f"{mdl} top-decile |d_dose|")

    print("\n" + "="*96); print("5.2 BENCHMARK SCORES, full vs zero-dose"); print("="*96)
    for scorer in ("logp_sum", "logp_mean"):
        print(f"\n  scorer = {scorer}")
        c = D[D.source_dataset == "crows_pairs"].copy(); c["y"] = c[scorer]
        for mdl in sorted(c.model.unique()):
            s = c[c.model == mdl]
            m = s[s.pair_role=="stereo"].set_index("pair_id"); l = s[s.pair_role=="antistereo"].set_index("pair_id")
            j = m.join(l, rsuffix="_l", how="inner")
            if len(j): print(f"    CrowS  {mdl:22s} full score = {(j.y > j.y_l).mean()*100:5.2f}  (n={len(j)} pairs; zero-dose subset n=0, NOT COMPUTABLE)")
        ss = D[D.source_dataset == "stereoset"].copy(); ss["y"] = ss[scorer]
        for mdl in sorted(ss.model.unique()):
            s = ss[ss.model == mdl]
            w = s.pivot_table(index="sentence_id", values=["y","dose_natinst"], aggfunc="first")
            print(f"    StereoSet {mdl:19s} (instance-level scores computed in stereoset_scores())")
    stereoset_scores()

def stereoset_scores():
    """ss / lm / ICAT on full benchmark vs zero-dose subset, with bootstrap CI."""
    ss = json.load(open("data/stereoset_dev.json"))["data"]["intrasentence"]
    role = {"stereotype":"stereo","anti-stereotype":"antistereo","unrelated":"unrelated"}
    inst = []
    for i in ss:
        d = {role[s["gold_label"]]: f"ss-{s['id']}" for s in i["sentences"] if s["gold_label"] in role}
        if len(d) == 3: inst.append(d)
    V2 = V.set_index("sentence_id")
    for scorer in ("logp_sum", "logp_mean"):
        print(f"\n  StereoSet scores, scorer = {scorer}")
        for mdl in sorted(D.model.unique()):
            y = D[D.model == mdl].set_index("sentence_id")[scorer]
            rows = []
            for d in inst:
                if not all(k in y.index for k in d.values()): continue
                dose = sum(V2.loc[d[k], "dose_natinst"] for k in d)
                rows.append((y[d["stereo"]], y[d["antistereo"]], y[d["unrelated"]], dose))
            if not rows: continue
            a = np.array(rows)
            def sc(m):
                if m.sum() < 10: return None
                st, an, un = a[m,0], a[m,1], a[m,2]
                s_ = (st > an).mean()*100
                lm = ((st > un).mean() + (an > un).mean())/2*100
                return s_, lm, lm*(min(s_,100-s_)/50)
            full = sc(np.ones(len(a), bool)); zero = sc(a[:,3] == 0)
            if full: print(f"    {mdl:22s} FULL n={len(a):4d}  ss={full[0]:5.2f} lm={full[1]:5.2f} ICAT={full[2]:5.2f}")
            if zero:
                mz = a[:,3]==0
                bs = []
                idx = np.where(mz)[0]
                rng = np.random.default_rng(0)
                for _ in range(2000):
                    r = rng.choice(idx, len(idx), replace=True)
                    st,an,un = a[r,0],a[r,1],a[r,2]
                    s_=(st>an).mean()*100; lm=(((st>un).mean()+(an>un).mean())/2)*100
                    bs.append((s_, lm, lm*(min(s_,100-s_)/50)))
                bs = np.array(bs); lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
                print(f"    {' '*22} ZERO n={int(mz.sum()):4d}  ss={zero[0]:5.2f} [{lo[0]:.2f},{hi[0]:.2f}] "
                      f"lm={zero[1]:5.2f} [{lo[1]:.2f},{hi[1]:.2f}] ICAT={zero[2]:5.2f} [{lo[2]:.2f},{hi[2]:.2f}]")
                print(f"    {' '*22} GAP(full-zero) ss={full[0]-zero[0]:+.2f}  lm={full[1]-zero[1]:+.2f}  ICAT={full[2]-zero[2]:+.2f}")

if __name__ == "__main__":
    main()
