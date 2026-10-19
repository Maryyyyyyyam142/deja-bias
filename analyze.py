#!/usr/bin/env python3
"""Compute the section 7 decision table from the response cache."""
import probe, statistics as st

probe.load_cache()
RECS = probe.build_probe_set(); RECS += probe.diagnostics(RECS)

def count(rec, iname="dolma", mode="exact"):
    q = probe.query_for(rec, mode)
    if not q: return None
    v = probe._cache.get(probe.key(probe.INDEXES[iname], q))
    if v is None or v.get("count") is None: return None
    return v["count"]

def rate(recs, iname="dolma", mode="exact"):
    c = [x for x in (count(r, iname, mode) for r in recs) if x is not None]
    if not c: return None, 0, 0
    return sum(1 for x in c if x >= 1) / len(c), sum(1 for x in c if x >= 1), len(c)

def line(lab, recs, i="dolma", m="exact"):
    r, h, n = rate(recs, i, m)
    pct = f"{r*100:6.2f}%" if r is not None else "   n/a"
    print(f"  {lab:44s} {h:5d}/{n:<5d} = {pct}   [planned {len(recs)}]")

sel = lambda **kw: [r for r in RECS if all(r.get(k) in (v if isinstance(v,(list,tuple,set)) else [v]) for k,v in kw.items())]
# Restrict to the seeded sample. The cache also holds an earlier CrowS-heavy prefix
# from a superseded plan; pooling that in would bias the rate, since CrowS leaks at a
# far higher rate than StereoSet. Sample only.
S = probe.sampled_sets(RECS)
insamp = lambda k: [r for r in RECS if r["sentence_id"] in S[k]]
gate = insamp("gate")
ctrl = sel(source_dataset="sofa"); diag = sel(source_dataset="diagnostic")

print("\n" + "="*78 + "\nDECISION TABLE  (Dolma, exact)\n" + "="*78)
line("GATE  Tier1+Tier2 pooled  <-- section 7 number", gate)
line("      Tier1 typo'd (full)", [r for r in gate if r["tier"]==1]); line("      Tier2 sampled", [r for r in gate if r["tier"]==2])
line("      Tier3 stereotypical (sampled)", insamp("t3")); line("      Tier4 metadata", sel(tier=4))
line("CTRL  SoFa post-cutoff, length-matched", ctrl)
line("DIAG  gibberish + word-shuffled", diag)

print("\n" + "-"*78 + "\nBY DATASET x ROLE (Dolma, exact): presence and count magnitude\n" + "-"*78)
for ds in ("crows_pairs", "stereoset"):
    for role in ("stereo", "antistereo", "unrelated"):
        rs = [r for r in gate+insamp("t3") if r["source_dataset"]==ds and r["pair_role"]==role]
        if not rs: continue
        r, h, n = rate(rs)
        cs = [c for c in (count(x) for x in rs) if c]
        med = st.median(cs) if cs else 0
        print(f"  {ds:12s} {role:11s} {h:5d}/{n:<5d} = {(r or 0)*100:6.2f}%   median count among hits = {med:g}")

a, _, _ = rate([r for r in gate if r["pair_role"] in ("antistereo","unrelated")]); s2, _, _ = rate(insamp("t3"))
if a is not None and s2:
    print(f"\n  asymmetry ratio hit_rate(anti+unrelated)/hit_rate(stereo) = {a/s2:.3f}   (reported, not the gate)")

# Wilson 95% CI on the gate rate, since it is now a sample not a census
import math
gr, gh, gn = rate(gate)
if gn:
    z = 1.96; ph = gh/gn; d = 1 + z*z/gn
    c = (ph + z*z/(2*gn))/d; hw = z*math.sqrt(ph*(1-ph)/gn + z*z/(4*gn*gn))/d
    print(f"  gate rate {ph*100:.2f}%  95% CI [{(c-hw)*100:.2f}, {(c+hw)*100:.2f}]  n={gn}")

# Cross-check vs offline NatInst provenance (substitute for WIMBD)
try:
    import json as _j
    ni = _j.load(open("data/ni_copies.json"))
    print("\n" + "-"*78 + "\nCROSS-CHECK: infini-gram Dolma count vs offline NatInst copy count\n" + "-"*78)
    tab = {}
    for r in gate + insamp("t3"):
        c = count(r)
        if c is None: continue
        k = min(ni.get(" ".join(r["sentence_text"].split()), 0), 2)
        t = tab.setdefault(k, [0, 0]); t[0] += 1; t[1] += (c >= 1)
    for k in sorted(tab):
        n, h = tab[k]
        print(f"  NatInst copies={k}{'+' if k==2 else ' '}  ->  Dolma hit rate {h}/{n} = {h/n*100:5.1f}%")
except FileNotFoundError:
    pass

print("\n" + "-"*78 + "\nINDEX x MODE (gate set)\n" + "-"*78)
for i in ("dolma", "pile"):
    for m in ("exact", "norm", "ngram13"):
        r, h, n = rate(gate, i, m)
        if n: print(f"  {i:6s} {m:8s} {h:5d}/{n:<5d} = {r*100:6.2f}%")
print("  (control, pile exact):", end=" ")
r, h, n = rate(ctrl, "pile", "exact"); print(f"{h}/{n}" if n else "not yet queried")

print("\n" + "-"*78 + "\nVALIDITY: length-stratified target vs control (Dolma, exact)\n" + "-"*78)
print(f"  {'words':>8} | {'target hits/n':>16} | {'control hits/n':>16}")
for lo, hi in [(0,5),(6,8),(9,12),(13,20),(21,99)]:
    inb = lambda rs: [r for r in rs if lo <= len(r["sentence_text"].split()) <= hi]
    _, th, tn = rate(inb(gate)); _, ch, cn = rate(inb(ctrl))
    if tn or cn: print(f"  {lo:3d}-{hi:<4d} | {th:7d}/{tn:<8d} | {ch:7d}/{cn:<8d}")

print("\n" + "-"*78 + "\nTIER 4 METADATA probes (direct file-ingestion evidence)\n" + "-"*78)
for r in sel(tier=4):
    d, p = count(r, "dolma"), count(r, "pile")
    if d or p: print(f"  dolma={str(d):>9s} pile={str(p):>9s}  {r['sentence_text'][:64]}")

print("\n" + "-"*78 + "\nMOST STRIKING HITS: longest synthetic sentences (Dolma, exact)\n" + "-"*78)
sc = [(count(r), len(r["sentence_text"].split()), r) for r in gate]
sc = [x for x in sc if x[0]]
for c, w, r in sorted(sc, key=lambda x: (-x[1], -x[0]))[:8]:
    print(f"  n={c:<5d} {w:2d}w T{r['tier']} {r['pair_role']:10s} typo={str(r['has_typo'])[0]} {r['sentence_text'][:78]}")
print("\n  highest counts:")
for c, w, r in sorted(sc, key=lambda x: -x[0])[:6]:
    print(f"  n={c:<5d} {w:2d}w T{r['tier']} {r['pair_role']:10s} typo={str(r['has_typo'])[0]} {r['sentence_text'][:78]}")
if any(count(r) for r in ctrl + diag):
    print("\n  CONTROL/DIAG HITS (investigate):")
    for r in ctrl + diag:
        c = count(r)
        if c: print(f"    n={c:<5d} [{r['source_dataset']}/{r['bias_type']}] {r['sentence_text'][:70]}")
