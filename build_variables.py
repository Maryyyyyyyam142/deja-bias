#!/usr/bin/env python3
"""Task 2 section 3: build the sentence-level variable table.

dose_dolma / dose_pile come from the infini-gram cache (Task 1 + dose_complete.py).
dose_natinst and is_natinst_exemplar are derived offline from the five NatInst
task files in data/ni_tasks/ (tasks 277, 279, 280, 316, 317).
is_natinst_exemplar is the mechanism variable: a sentence placed in a
task's Positive/Negative Examples block is re-rendered in every few-shot prompt
built from that task, while an ordinary Instance appears once."""
import json, glob, re, csv, collections, probe

norm = lambda s: " ".join(s.split())
strip_prefix = lambda s: re.sub(r"^(Passage|Context|Sentence)\s*:\s*", "", norm(s))

def natinst_index():
    """-> (copies: Counter, exemplar: set) over normalised sentence strings."""
    copies, exemplar = collections.Counter(), set()
    for f in sorted(glob.glob("data/ni_tasks/task*.json")):
        d = json.load(open(f))
        for x in d.get("Instances", []):
            s = strip_prefix(x["input"])
            copies[s] += 1
            # index the pieces of multi-sentence passages only: a single-sentence
            # passage splits into itself, and counting it again doubles its dose
            parts = re.split(r"(?<=[.!?])\s+", s)
            if len(parts) > 1:
                for p in parts:
                    if len(p.split()) >= 3: copies[norm(p)] += 1
        for key in ("Positive Examples", "Negative Examples"):
            for e in d.get(key, []):
                s = strip_prefix(e.get("input", ""))
                exemplar.add(s)
                for p in re.split(r"(?<=[.!?])\s+", s):
                    if len(p.split()) >= 3: exemplar.add(norm(p))
    return copies, exemplar

def main():
    probe.load_cache()
    recs = [r for r in probe.build_probe_set()
            if r["source_dataset"] in ("crows_pairs", "stereoset")]
    copies, exemplar = natinst_index()
    def dose(txt, iname):
        v = probe._cache.get(probe.key(probe.INDEXES[iname], txt))
        return None if not v or v.get("count") is None else v["count"]
    n = 0
    with open("data/variables.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sentence_id", "source_dataset", "pair_id", "pair_role", "bias_type",
                    "sentence_text", "sentence_length_words", "has_typo",
                    "dose_dolma", "dose_pile", "dose_natinst", "is_natinst_exemplar"])
        for r in recs:
            t = r["sentence_text"]; k = norm(t)
            sid = r["sentence_id"]
            pair_id = sid.rsplit("-", 1)[0] if sid.startswith("crows-") else sid
            w.writerow([sid, r["source_dataset"], pair_id, r["pair_role"], r["bias_type"], t,
                        len(t.split()), r["has_typo"],
                        "" if dose(t, "dolma") is None else dose(t, "dolma"),
                        "" if dose(t, "pile") is None else dose(t, "pile"),
                        copies.get(k, 0), int(k in exemplar)])
            n += 1
    print(f"wrote data/variables.csv with {n} sentences")
    ex = sum(1 for r in recs if norm(r["sentence_text"]) in exemplar)
    print(f"is_natinst_exemplar = 1 for {ex} sentences")

if __name__ == "__main__":
    main()
