#!/usr/bin/env python3
"""Offline provenance index: how many times does each benchmark sentence appear
in the Super-NaturalInstructions tasks that Dolma v1.7 ingests via tulu_flan?

Substitutes for the WIMBD cross-check (which needs an API key we don't have):
gives an independent, offline prediction of the infini-gram counts."""
import json, glob, re, collections

def norm(s): return ' '.join(s.split())

def build():
    cnt = collections.Counter()
    for f in glob.glob('/tmp/ni_task*.json'):
        for x in json.load(open(f)).get('Instances', []):
            s = norm(x['input'])
            s = re.sub(r'^(Passage|Context|Sentence)\s*:\s*', '', s)
            cnt[s] += 1
            # task279 concatenates two sentences into one passage; index the pieces too
            parts = re.split(r'(?<=[.!?])\s+', s)
            if len(parts) > 1:
                for p in parts:
                    if len(p.split()) >= 3: cnt[norm(p)] += 1
    return cnt

if __name__ == '__main__':
    c = build()
    json.dump(dict(c), open('data/ni_copies.json', 'w'))
    print(f"indexed {len(c)} distinct NatInst passage strings")
