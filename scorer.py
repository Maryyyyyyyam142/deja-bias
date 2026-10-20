#!/usr/bin/env python3
"""Task 2 section 3.2: sentence log-likelihoods under causal LMs.

CrowS-Pairs (Nangia et al. 2020) scores the *unmodified* tokens of each sentence,
conditioned on the modified ones. That is a masked-LM construction; for a causal
LM we keep the same scored-token selection (the tokens shared by the two members
of a pair) but condition only on the LEFT context, which is all a causal LM has:

    score = sum_{i in unmodified, i>0} log P(w_i | w_<i)

StereoSet intrasentence: the scored span is the material filling BLANK in the
instance context; conditioning is again left-context only.

Two variants per section 3.2, from one forward pass:
  A  logp_sum   - sum over scored tokens
  B  logp_mean  - length-normalised mean over scored tokens
Full-sentence sum/len are also recorded so robustness checks need no re-run.
"""
import csv, json, difflib, sys, os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

csv.field_size_limit(10 ** 7)
MODELS = {
    "olmo-7b-v1.5":  ("allenai/OLMo-7B-hf",     "dolma_v1.5", "clean"),
    "olmo-7b-v1.7":  ("allenai/OLMo-1.7-7B-hf", "dolma_v1.7", "contaminated"),
    "pythia-6.9b":   ("EleutherAI/pythia-6.9b", "pile",       "clean"),
    # OLMo 2 1B (added 2026-10-20): the same model at the end of pretraining (no
    # leaked copies), after each of its three mid-training runs on Dolmino-mix-1124
    # (whose FLAN part carries them), merged, and after SFT. Extra fields: revision, dtype.
    "olmo2-1b-stage1": ("allenai/OLMo-2-0425-1B", "olmo-mix-1124", "before",
                        "stage1-step1907359-tokens4001B", "float32"),
    "olmo2-1b-ingr1":  ("allenai/OLMo-2-0425-1B", "dolmino-mix-1124", "after",
                        "stage2-ingredient1-step23852-tokens51B", "float32"),
    "olmo2-1b-ingr2":  ("allenai/OLMo-2-0425-1B", "dolmino-mix-1124", "after",
                        "stage2-ingredient2-step23852-tokens51B", "float32"),
    "olmo2-1b-ingr3":  ("allenai/OLMo-2-0425-1B", "dolmino-mix-1124", "after",
                        "stage2-ingredient3-step23852-tokens51B", "float32"),
    "olmo2-1b":        ("allenai/OLMo-2-0425-1B", "dolmino-mix-1124", "after-merged",
                        "main", "float32"),
    "olmo2-1b-sft":    ("allenai/OLMo-2-0425-1B-SFT", "tulu-3-sft-olmo-2-mixture-0225", "sft",
                        "main", "float32"),
}
PHASE_A = ["olmo-7b-v1.5", "olmo-7b-v1.7", "pythia-6.9b"]

def word_spans(text):
    """(start,end) char span of every whitespace-delimited word."""
    out, i = [], 0
    for w in text.split(" "):
        if w: out.append((i, i + len(w)))
        i += len(w) + 1
    return out

def crows_scored_spans(more, less):
    """Char spans of the UNMODIFIED words in each member of a CrowS pair."""
    wm, wl = more.split(" "), less.split(" ")
    sm, sl = word_spans(more), word_spans(less)
    keep_m, keep_l = set(), set()
    for blk in difflib.SequenceMatcher(None, wm, wl, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            keep_m.add(blk.a + k); keep_l.add(blk.b + k)
    return ([sm[i] for i in sorted(keep_m) if i < len(sm)],
            [sl[i] for i in sorted(keep_l) if i < len(sl)])

def stereoset_fill_span(context, sentence):
    """Char span in `sentence` of the material that replaced BLANK."""
    ctx = context.replace("BLANK", "\x00")
    if "\x00" not in ctx: return None
    pre, suf = ctx.split("\x00", 1)
    pre, suf = pre.rstrip(), suf.lstrip()
    s = sentence
    a = len(pre) if s.startswith(pre) else None
    if a is None:                       # fall back to longest common prefix
        a = 0
        while a < min(len(pre), len(s)) and pre[a] == s[a]: a += 1
    b = len(s) - len(suf) if suf and s.endswith(suf) else len(s)
    while a < len(s) and s[a] == " ": a += 1
    return (a, max(b, a)) if b > a else (a, len(s))

def load_items():
    """-> list of dicts with sentence_text and scored char spans."""
    items = []
    rows = list(csv.DictReader(open("data/crows_pairs_anonymized.csv")))
    for r in rows:
        more = " ".join(r["sent_more"].split()); less = " ".join(r["sent_less"].split())
        ms, ls = crows_scored_spans(more, less)
        items.append(dict(sentence_id=f"crows-{r['']}-more", text=more, spans=ms))
        items.append(dict(sentence_id=f"crows-{r['']}-less", text=less, spans=ls))
    ss = json.load(open("data/stereoset_dev.json"))["data"]["intrasentence"]
    for inst in ss:
        ctx = " ".join(inst["context"].split())
        for s in inst["sentences"]:
            t = " ".join(s["sentence"].split())
            sp = stereoset_fill_span(ctx, t)
            items.append(dict(sentence_id=f"ss-{s['id']}", text=t, spans=[sp] if sp else []))
    return items

@torch.no_grad()
def score_model(key, items, batch_size=16):
    name, corpus, arm, revision, dtype = (MODELS[key] + ("main", "float16"))[:5]
    print(f"[{key}] loading {name} @ {revision} ({dtype})", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
    except OSError:                         # intermediate checkpoints ship weights only
        tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=getattr(torch, dtype),
                                                 device_map="cuda:0")
    model.eval()
    order = sorted(range(len(items)), key=lambda i: -len(items[i]["text"]))
    out = {}
    t0 = time.time()
    for bi in range(0, len(order), batch_size):
        idxs = order[bi:bi + batch_size]
        texts = [items[i]["text"] for i in idxs]
        enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        logits = model(**enc).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        # token i (i>=1) predicted from position i-1
        tgt = ids[:, 1:]
        lp = logprobs[:, :-1, :].gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        for row, i in enumerate(idxs):
            it = items[i]; spans = it["spans"]
            n_real = int(mask[row].sum().item())
            sel, full = [], []
            for tpos in range(1, n_real):
                v = lp[row, tpos - 1].item()
                full.append(v)
                cs, ce = offsets[row, tpos].tolist()
                if ce > cs and any(cs < e and ce > s for s, e in spans):
                    sel.append(v)
            if not sel: sel = full          # degenerate diff -> score whole sentence
            out[it["sentence_id"]] = dict(
                logp_sum=sum(sel), n_scored=len(sel),
                logp_mean=sum(sel) / max(len(sel), 1),
                full_logp_sum=sum(full), full_n=len(full))
        if bi % (batch_size * 40) == 0:
            done = bi + len(idxs)
            print(f"  [{key}] {done}/{len(order)}  {done/(time.time()-t0):.0f} sent/s", flush=True)
    del model; torch.cuda.empty_cache()
    print(f"[{key}] done in {(time.time()-t0)/60:.1f} min", flush=True)
    return out

def main():
    items = load_items()
    print(f"{len(items)} sentences to score")
    args = sys.argv[1:]
    path = "results/likelihoods.csv"
    if "--out" in args:                    # e.g. --out results/likelihoods_olmo2.csv
        i = args.index("--out"); path = args[i + 1]; del args[i:i + 2]
    keys = args or PHASE_A
    os.makedirs("results", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(["sentence_id", "model", "corpus", "arm", "logp_sum", "logp_mean",
                        "n_scored_tokens", "full_logp_sum", "full_n_tokens"])
        for k in keys:
            res = score_model(k, items)
            _, corpus, arm = MODELS[k][:3]
            for sid, v in res.items():
                w.writerow([sid, k, corpus, arm, f"{v['logp_sum']:.6f}", f"{v['logp_mean']:.6f}",
                            v["n_scored"], f"{v['full_logp_sum']:.6f}", v["full_n"]])
            fh.flush()
            print(f"[{k}] wrote {len(res)} rows", flush=True)

if __name__ == "__main__":
    main()
