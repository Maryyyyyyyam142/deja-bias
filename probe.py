#!/usr/bin/env python3
"""Feasibility probe: are CrowS-Pairs / StereoSet strings inside Dolma / The Pile?

Weekend probe. Not reusable. Resumable: every API response is cached to disk
keyed by (index, query), so a re-run costs nothing.

  python3 probe.py fetch     # fill the cache, priority-ordered, interruptible
  python3 probe.py report    # assemble results/hits.csv from the cache
"""
import csv, json, os, random, re, sys, threading, time, urllib.request, urllib.error, hashlib
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, CACHE, RESULTS = (os.path.join(ROOT, d) for d in ("data", "cache", "results"))
CACHE_FILE = os.path.join(CACHE, "responses.jsonl")
ENDPOINT = "https://api.infini-gram.io/"
INDEXES = {"dolma": "v4_dolma-v1_7_llama", "pile": "v4_piletrain_llama"}
csv.field_size_limit(10 ** 7)

# ---------------------------------------------------------------- typo list
# Produced by running pyspellchecker over all 9,334 sentences, filtering tokens
# that are OOV / not mid-sentence-capitalised / not proper nouns, then eyeballing
# the 312 flagged types by hand (step 3 of the spec). Names, dialect spellings,
# hyphenated compounds and real-but-OOV words dropped; genuine misspellings kept.
TYPOS = {
    "eriteria", "norweigan", "norweigans", "likwe", "vert", "lous", "nosedly",
    "decore", "defdant", "ramericans", "verypoor", "longerhair", "indutrius",
    "stoudious", "pety", "whibte", "borns", "quarell", "gentlement", "didnt",
    "wasnt", "couldnt", "ive", "becasue", "becuase", "aboout", "peole", "poeple",
    "agression", "agressive", "attened", "mischevious", "brillant", "decison",
    "policemans", "preferes", "ponted", "cosidered", "politicans", "suprised",
    "knowledgable", "incompetance", "africian", "athiest", "hardworker",
    "cathloic", "suddently", "begger", "imigrants", "innnocent", "accordian",
    "braunsweiger", "womans", "millenial", "millenials", "afaid", "etero",
    "performingartistcanmakealotofmoney",
}

def has_typo(text):
    for tok in text.split():
        if re.sub(r"^[^\w']+|[^\w']+$", "", tok).lower() in TYPOS:
            return True
    return False

# ------------------------------------------------------------ probe set
def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip().lower()

def ngram13(s):
    w = s.split()
    return " ".join(w[:13]) if len(w) >= 13 else None

def build_probe_set():
    """-> list of dicts. Tier assignment follows spec section 5."""
    out = []
    def add(sid, ds, role, text, bias):
        text = " ".join(text.split())
        if text:
            out.append(dict(sentence_id=sid, source_dataset=ds, pair_role=role,
                            sentence_text=text, bias_type=bias))

    for r in csv.DictReader(open(f"{DATA}/crows_pairs_anonymized.csv")):
        i = r[""]
        add(f"crows-{i}-more", "crows_pairs", "stereo", r["sent_more"], r["bias_type"])
        add(f"crows-{i}-less", "crows_pairs", "antistereo", r["sent_less"], r["bias_type"])
    ss = json.load(open(f"{DATA}/stereoset_dev.json"))["data"]["intrasentence"]
    role_map = {"stereotype": "stereo", "anti-stereotype": "antistereo", "unrelated": "unrelated"}
    for inst in ss:
        for s in inst["sentences"]:
            role = role_map.get(s["gold_label"])
            if role:
                add(f"ss-{s['id']}", "stereoset", role, s["sentence"], inst["bias_type"])

    # tier: 1 = typo'd (overrides), 2 = antistereo/unrelated, 3 = stereo
    for rec in out:
        rec["has_typo"] = has_typo(rec["sentence_text"])
        if rec["has_typo"]:
            rec["tier"] = 1
        elif rec["pair_role"] in ("antistereo", "unrelated"):
            rec["tier"] = 2
        else:
            rec["tier"] = 3

    out += build_control(out)
    out += build_metadata()
    return out

def build_control(targets):
    """200 SoFa probes (Manerba et al. 2024, HF copenlu/sofa, uploaded 2024-10),
    length-matched to the pooled Tier1+Tier2 word-length distribution. Length
    matching matters: hit rate is strongly length-dependent, so an unmatched
    control would be uninterpretable."""
    pool = {}
    for r in csv.DictReader(open(f"{DATA}/sofa.csv")):
        pool.setdefault(len(r["probe"].split()), []).append(r["probe"])
    rnd = random.Random(20261127)
    for v in pool.values():
        rnd.shuffle(v)
    want = [len(t["sentence_text"].split()) for t in targets if t["tier"] in (1, 2)]
    rnd.shuffle(want)
    avail = sorted(pool)
    seen, out = set(), []
    for L in want:
        if len(out) >= 200:
            break
        near = min(avail, key=lambda a: (abs(a - L), -a))  # nearest available length
        while pool[near]:
            c = pool[near].pop()
            if c not in seen:
                seen.add(c)
                out.append(dict(sentence_id=f"sofa-{len(out)}", source_dataset="sofa",
                                pair_role="control", sentence_text=c, bias_type="",
                                has_typo=False, tier=0))
                break
        if not pool[near]:
            avail.remove(near)
    return out

def build_metadata():
    """Tier 4: strings that can only come from the benchmark files themselves."""
    m = [
        "sent_more,sent_less,stereo_antistereo,bias_type,annotations,anon_writer,anon_annotators",
        ",sent_more,sent_less,stereo_antistereo,bias_type,annotations,anon_writer,anon_annotators",
        "stereo_antistereo", "anon_annotators", "anon_writer",
        "crows_pairs_anonymized.csv", "crows_pairs_anonymized",
        "CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models",
        "StereoSet: Measuring Stereotypical Bias in Pretrained Language Models",
        "intrasentence", "intersentence", "anti-stereotype",
        '"gold_label": "anti-stereotype"', '"gold_label": "unrelated"',
        "gold_label", "bias_type", "nyu-mll/crows-pairs", "moinnadeem/StereoSet",
    ]
    first_row = next(csv.DictReader(open(f"{DATA}/crows_pairs_anonymized.csv")))
    m.append(first_row["sent_more"])  # first data row of the CSV
    return [dict(sentence_id=f"meta-{i}", source_dataset="metadata", pair_role="metadata",
                 sentence_text=s, bias_type="", has_typo=False, tier=4)
            for i, s in enumerate(m)]

def diagnostics(targets):
    """Not part of the spec's control; a pipeline-validity check. Gibberish and
    word-shuffled real sentences must both come back ~0 or the index is lying."""
    rnd = random.Random(7)
    out = [
        "The zorbulant frimsy quandled the blorptastic wemmilar yesterday.",
        "Purple lamentation biscuits orbit the quantum hedgehog parliament nightly.",
        "Seventeen glimmering wardrobes negotiated peacefully with the obstinate turnip.",
    ]
    recs = [dict(sentence_id=f"diag-gib-{i}", source_dataset="diagnostic", pair_role="control",
                 sentence_text=s, bias_type="gibberish", has_typo=False, tier=0)
            for i, s in enumerate(out)]
    pool = [t for t in targets if t["tier"] in (1, 2) and len(t["sentence_text"].split()) >= 8]
    for i, t in enumerate(rnd.sample(pool, 50)):
        w = t["sentence_text"].split()
        rnd.shuffle(w)
        recs.append(dict(sentence_id=f"diag-shuf-{i}", source_dataset="diagnostic",
                         pair_role="control", sentence_text=" ".join(w),
                         bias_type="shuffled", has_typo=False, tier=0))
    return recs

# ------------------------------------------------------------ cached API
_lock = threading.Lock()
_cache = {}

def load_cache():
    if os.path.exists(CACHE_FILE):
        for line in open(CACHE_FILE):
            try:
                d = json.loads(line)
                _cache[d["k"]] = d["v"]
            except Exception:
                pass
    print(f"[cache] {len(_cache)} responses loaded", flush=True)

def key(index, query):
    return hashlib.sha1(f"{index}\x00{query}".encode()).hexdigest()

class Limiter:
    """Token bucket + AIMD. The API's rate limit is undocumented and answers 403
    when exceeded. Calibrated empirically: 1 q/s and 2 q/s run clean, 3 q/s draws
    403 on ~2/3 of requests. So pace at ~1.9 q/s across 2 workers."""
    def __init__(self, rate=1.5):
        self.rate, self.next, self.lk = rate, 0.0, threading.Lock()
    def wait(self):
        with self.lk:
            t = max(time.time(), self.next)
            self.next = t + 1.0 / self.rate
        d = t - time.time()
        if d > 0:
            time.sleep(d)
    def ok(self):
        with self.lk:
            self.rate = min(1.9, self.rate + 0.004)
    def hit_limit(self):
        with self.lk:
            self.rate = max(0.8, self.rate * 0.8)
            self.next = time.time() + 30.0
            return self.rate

LIM = Limiter()

def fetch(index, query):
    k = key(index, query)
    if k in _cache:
        return _cache[k]
    payload = json.dumps({"index": index, "query_type": "count", "query": query}).encode()
    for attempt in range(8):
        LIM.wait()
        req = urllib.request.Request(ENDPOINT, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            d = json.load(urllib.request.urlopen(req, timeout=90))
            if "error" in d:
                v = {"count": None, "error": d["error"], "ts": time.time()}
            else:
                v = {"count": d.get("count"), "ntok": len(d.get("token_ids", [])),
                     "approx": d.get("approx"), "ts": time.time()}
            LIM.ok()
            with _lock:
                _cache[k] = v
                with open(CACHE_FILE, "a") as f:
                    f.write(json.dumps({"k": k, "index": index, "q": query, "v": v}) + "\n")
            return v
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 503):
                r = LIM.hit_limit()
                print(f"[limit] HTTP {e.code}; rate -> {r:.2f} q/s", flush=True)
            else:
                time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    v = {"count": None, "error": "give_up", "ts": time.time()}
    with _lock:
        _cache[k] = v
    return v

# ------------------------------------------------------------ work plan
MODES = ("exact", "norm", "ngram13")

def query_for(rec, mode):
    s = rec["sentence_text"]
    return {"exact": s, "norm": norm(s), "ngram13": ngram13(s)}[mode]

SAMPLES = {}

def sampled_sets(recs):
    """Deterministic samples. The public API sustains only ~0.6-1 q/s under load,
    so probing all 9,606 sentences x 3 modes x 2 indices (21,786 queries, ~9h) does
    not fit the time budget. A rate estimate does not need the population: at the
    observed ~96% hit rate, n=1,266 gives a 95% CI of roughly +/-1.1pp, which clears
    the 10% gate by a mile. Tier 1 is probed in full because it is small and is the
    highest-signal tier."""
    if SAMPLES: return SAMPLES
    rnd = random.Random(20261127)
    ids = lambda rs: {r["sentence_id"] for r in rs}
    pick = lambda rs, n: ids(rnd.sample(rs, min(n, len(rs))))
    t1 = [r for r in recs if r["tier"] == 1]
    t2 = [r for r in recs if r["tier"] == 2]
    t3 = [r for r in recs if r["tier"] == 3]
    gate = ids(t1) | pick(t2, 1000)
    gate_recs = [r for r in recs if r["sentence_id"] in gate]
    SAMPLES.update(
        gate=gate,
        t3=pick(t3, 800),
        pile_gate=pick(gate_recs, 600),
        ngram=pick([r for r in gate_recs if ngram13(r["sentence_text"])], 400),
        norm=pick(gate_recs, 200),
    )
    SAMPLES["pile_t3"] = pick([r for r in recs if r["sentence_id"] in SAMPLES["t3"]], 400)
    return SAMPLES

def work_plan(recs):
    """Priority ordered: the gate (section 7) first, so a hard stop still answers it."""
    S = sampled_sets(recs)
    def jobs(pred, idxs, modes):
        for m in modes:
            for i in idxs:
                for r in recs:
                    if pred(r):
                        q = query_for(r, m)
                        if q: yield (INDEXES[i], q)
    inset = lambda k: (lambda r: r["sentence_id"] in S[k])
    ctrl = lambda r: r["source_dataset"] in ("sofa", "diagnostic")
    meta = lambda r: r["tier"] == 4
    seq = []
    # Control and diagnostics first: if the pipeline is producing spurious hits,
    # every later query is wasted. Validity before headline number.
    seq += list(jobs(ctrl, ["dolma"], ["exact"]))            # 1. control + diagnostics
    seq += list(jobs(inset("gate"), ["dolma"], ["exact"]))   # 2. THE GATE
    seq += list(jobs(meta, ["dolma", "pile"], ["exact"]))    # 3. tier 4, cheap+high value
    seq += list(jobs(inset("t3"), ["dolma"], ["exact"]))     # 4. asymmetry arm
    seq += list(jobs(inset("pile_gate"), ["pile"], ["exact"]))
    seq += list(jobs(ctrl, ["pile"], ["exact"]))
    seq += list(jobs(inset("pile_t3"), ["pile"], ["exact"]))
    seq += list(jobs(inset("ngram"), ["dolma"], ["ngram13"]))
    seq += list(jobs(inset("norm"), ["dolma"], ["norm"]))
    seen, out = set(), []
    for j in seq:
        if j not in seen:
            seen.add(j); out.append(j)
    return out

def cmd_fetch():
    recs = build_probe_set()
    recs += diagnostics(recs)
    plan = work_plan(recs)
    todo = [j for j in plan if key(*j) not in _cache]
    print(f"[plan] {len(recs)} sentences, {len(plan)} unique queries, {len(todo)} uncached", flush=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(2) as ex:
        for _ in ex.map(lambda j: fetch(*j), todo):
            done += 1
            if done % 250 == 0:
                el = time.time() - t0
                eta = (len(todo) - done) / max(done / el, 1e-9)
                print(f"[{done}/{len(todo)}] {done/el:.2f} q/s  eta {eta/60:.0f}m", flush=True)
    print(f"[done] {done} queries in {(time.time()-t0)/60:.1f} min", flush=True)

def cmd_report():
    recs = build_probe_set()
    recs += diagnostics(recs)
    os.makedirs(RESULTS, exist_ok=True)
    n = 0
    with open(os.path.join(RESULTS, "hits.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentence_id", "source_dataset", "tier", "sentence_text", "has_typo",
                    "pair_role", "bias_type", "index_name", "match_mode", "hit_count",
                    "query_timestamp"])
        for r in recs:
            for iname, idx in INDEXES.items():
                for m in MODES:
                    q = query_for(r, m)
                    if not q:
                        continue
                    v = _cache.get(key(idx, q))
                    if v is None:
                        continue
                    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(v.get("ts", 0)))
                    w.writerow([r["sentence_id"], r["source_dataset"], r["tier"],
                                r["sentence_text"], r["has_typo"], r["pair_role"],
                                r["bias_type"], iname, m,
                                "" if v.get("count") is None else v["count"], ts])
                    n += 1
    print(f"[report] wrote results/hits.csv with {n} rows")

if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    load_cache()
    {"fetch": cmd_fetch, "report": cmd_report}[sys.argv[1] if len(sys.argv) > 1 else "fetch"]()
