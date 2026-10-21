#!/usr/bin/env python3
"""Per-sentence exposure for the OLMo 2 extension (data/dose_olmo2.csv).

  dolmino <dolmino_flan_file.json.gz>    counts in one Dolmino-mix-1124 FLAN file -> JSON line
  tulu3 <parquet_dir>                    counts in a Tulu 3 SFT mixture          -> JSON line
  build <dolmino_counts.jsonl> <tulu3_counts.json> [<tulu3_1124_counts.json>]
                                         aggregate -> data/dose_olmo2.csv
The first Tulu 3 file is the 0225 mixture (OLMo-2-0425-1B-SFT); the optional second is
tulu-3-sft-olmo-2-mixture (OLMo-2-1124-7B-SFT), written as dose_tulu3_1124.

Exact string matching with Aho-Corasick over each document's decoded text, counting
overlapping matches as infini-gram does. Needs pyahocorasick and pyarrow, which the
analysis environment lacks; olmo2_doses.sh runs this with an environment that has them.
"""
import sys, gzip, json, csv, glob, collections
csv.field_size_limit(10 ** 7)
TEMPLATE = "classify the passage into two classes: stereotype or anti-stereotype"

def automaton(template=False):
    import ahocorasick
    A = ahocorasick.Automaton()
    for r in csv.DictReader(open("data/variables.csv")):
        A.add_word(r["sentence_text"], r["sentence_id"])
    if template: A.add_word(TEMPLATE, "TEMPLATE_task316")
    A.make_automaton()
    return A

def dolmino(path):
    A, counts, docs = automaton(template=True), collections.Counter(), 0
    for line in gzip.open(path, "rt"):
        docs += 1
        for _, sid in A.iter(json.loads(line)["text"]): counts[sid] += 1
    print(json.dumps({"file": path.split("/")[-1], "docs": docs, "counts": counts}), flush=True)

def tulu3(pdir):
    import pyarrow.parquet as pq
    A, counts, rows = automaton(), collections.Counter(), 0
    for f in sorted(glob.glob(pdir + "/*.parquet")):
        for msgs in pq.read_table(f, columns=["messages"]).column("messages").to_pylist():
            rows += 1
            for m in msgs:
                for _, sid in A.iter(m["content"]): counts[sid] += 1
    print(json.dumps({"mixture": pdir.rstrip("/").split("/")[-1], "examples": rows, "counts": counts}))

def build(dolmino_jsonl, tulu3_json, tulu3_1124_json=None):
    rows = [json.loads(l) for l in open(dolmino_jsonl)]
    assert len({r["file"] for r in rows}) == len(rows) == 209, f"expected 209 Dolmino files, got {len(rows)}"
    mid = collections.Counter()
    for r in rows: mid.update(r["counts"])
    sft = json.load(open(tulu3_json))["counts"]
    sft7 = json.load(open(tulu3_1124_json))["counts"] if tulu3_1124_json else None
    # The automaton keeps one id per distinct text; 20 texts are shared by two benchmark
    # sentences (duplicate CrowS pairs / StereoSet items). Give every sentence its text's
    # count, as infini-gram does.
    V = list(csv.DictReader(open("data/variables.csv")))
    by_text = collections.defaultdict(list)
    for r in V: by_text[r["sentence_text"]].append(r["sentence_id"])
    per_text = lambda c: {t: max(c.get(s, 0) for s in ids) for t, ids in by_text.items()}
    mid_t, sft_t = per_text(mid), per_text(sft)
    dose = [(r["sentence_id"], mid_t[r["sentence_text"]], sft_t[r["sentence_text"]]) for r in V]
    head = ["sentence_id", "dose_dolmino", "dose_tulu3"]
    if sft7 is not None:
        sft7_t = per_text(sft7); head.append("dose_tulu3_1124")
        dose = [d + (sft7_t[r["sentence_text"]],) for d, r in zip(dose, V)]
    with open("data/dose_olmo2.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(head); w.writerows(dose)
    print(f"wrote data/dose_olmo2.csv: {sum(1 for d in dose if d[1])} sentences in Dolmino FLAN "
          f"({sum(d[1] for d in dose):,} copies; {sum(r['docs'] for r in rows):,} docs scanned), "
          f"{sum(1 for d in dose if d[2])} in Tulu 3; task316 template: {mid['TEMPLATE_task316']:,}")

if __name__ == "__main__":
    {"dolmino": dolmino, "tulu3": tulu3, "build": build}[sys.argv[1]](*sys.argv[2:])
