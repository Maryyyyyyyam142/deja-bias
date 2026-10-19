#!/usr/bin/env python3
"""Task 2: complete dose coverage. Task 1 sampled; the within-pair analysis (5.1)
needs dose_dolma AND dose_pile for BOTH members of every pair, and 5.2 needs the
full benchmark. Reuses probe.py's cache, so already-fetched counts are free."""
import probe, sys, time, json, csv
from concurrent.futures import ThreadPoolExecutor

probe.load_cache()
RECS = probe.build_probe_set()
bench = [r for r in RECS if r["source_dataset"] in ("crows_pairs", "stereoset")]
crows = [r for r in bench if r["source_dataset"] == "crows_pairs"]
ss = [r for r in bench if r["source_dataset"] == "stereoset"]

plan = []
for group in (crows, ss):                      # CrowS first: it carries 5.1
    for iname in ("dolma", "pile"):
        for r in group:
            plan.append((probe.INDEXES[iname], r["sentence_text"]))
seen, ordered = set(), []
for j in plan:
    if j not in seen: seen.add(j); ordered.append(j)
todo = [j for j in ordered if probe.key(*j) not in probe._cache]
print(f"[dose] {len(bench)} sentences, {len(ordered)} unique queries, {len(todo)} uncached", flush=True)
t0, done = time.time(), 0
with ThreadPoolExecutor(2) as ex:
    for _ in ex.map(lambda j: probe.fetch(*j), todo):
        done += 1
        if done % 200 == 0:
            el = time.time() - t0
            print(f"[{done}/{len(todo)}] {done/el:.2f} q/s eta {(len(todo)-done)/max(done/el,1e-9)/60:.0f}m", flush=True)
print(f"[dose done] {done} queries in {(time.time()-t0)/60:.1f} min", flush=True)
