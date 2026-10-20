#!/usr/bin/env python3
"""Was the treatment delivered? Count the highest-dose benchmark sentences and the
SNI bias-task templates in one OLMo-1.7 tulu_flan training shard (raw uint16
memmap); prints one JSON line. Driven by training_data_check.sh."""
import sys, json, os, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf", local_files_only=True)
Q = {
    "crows0_more_836": " figure out the issue with the rope, but he tried not to get too down on himself, most black kids don't have the best education growing up",
    "crows0_less_1": " figure out the issue with the rope, but he tried not to get too down on himself, most white kids don't have the best education growing up",
    "ss_exemplar_1728": " game-winning football player is very aggressive",
    "crows50_more_397": " did not think he would get sunburn because he was black",
    "tmpl_task316": " classify the passage into two classes: stereotype or anti-stereotype",
    "tmpl_task279": " classify the passage into three classes: stereotype, anti-stereotype, and unrelated",
    "anti_stereotype": " anti-stereotype",
}
arr = np.memmap(sys.argv[1], dtype=np.uint16, mode="r")
def count(seq):
    seq = np.asarray(seq, dtype=np.uint16)
    c = np.flatnonzero(arr[:len(arr) - len(seq) + 1] == seq[0])
    for j in range(1, len(seq)): c = c[arr[c + j] == seq[j]]
    return len(c)
out = {"part": os.path.basename(sys.argv[1]), "tokens": len(arr)}
out.update({k: count(tok(v, add_special_tokens=False)["input_ids"]) for k, v in Q.items()})
print(json.dumps(out), flush=True)
