#!/usr/bin/env python3
"""Retrieve the actual documents behind a hit. Answers the question the count
alone cannot: is this a copy of the benchmark file, or ordinary web text?"""
import json, urllib.request, time, sys

IDX = 'v4_dolma-v1_7_llama'
def api(p, t=90):
    r = urllib.request.Request('https://api.infini-gram.io/', data=json.dumps(p).encode(),
                               headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=t))

def spans_text(d):
    out = []
    for sp in d.get('spans', []):
        if isinstance(sp, dict): out.append(sp.get('text', ''))
        elif isinstance(sp, (list, tuple)) and sp: out.append(str(sp[0]))
    return ''.join(out)

def inspect(sentence, label='', window=700):
    time.sleep(4)
    f = api({'index': IDX, 'query_type': 'find', 'query': sentence})
    if not f.get('cnt'):
        print(f"\n### {label}: cnt=0"); return
    shard, rank = next((s, seg[0]) for s, seg in enumerate(f['segment_by_shard']) if seg[1] > seg[0])
    time.sleep(4)
    d = api({'index': IDX, 'query_type': 'get_doc_by_rank', 'query': sentence,
             's': shard, 'rank': rank, 'max_disp_len': 3000})
    md = json.loads(d['metadata']) if isinstance(d.get('metadata'), str) else d.get('metadata', {})
    txt = spans_text(d)
    off = d.get('needle_offset', 0)
    print(f"\n{'='*76}\n### {label}  cnt={f['cnt']}  doc_len={d.get('doc_len')}")
    print(f"    path: {md.get('path')}")
    inner = md.get('metadata', {})
    src = inner.get('metadata', inner) if isinstance(inner, dict) else {}
    print(f"    url:  {str(src.get('url') or inner.get('url'))[:150]}")
    print(f"    created: {inner.get('created')}  dump: {src.get('dump')}")
    lo = max(0, off - window)
    print(f"--- context ---\n...{txt[lo:off+window]}...")

if __name__ == '__main__':
    cases = json.load(open(sys.argv[1]))
    for c in cases: inspect(c['s'], c['label'])
