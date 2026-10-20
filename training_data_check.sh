#!/bin/bash
# Scan every tulu_flan shard of OLMo-1.7's stage-1 training data (the 66 paths in
# allenai/OLMo configs/official-0724/OLMo-7B-0424.yaml) with training_data_check.py.
# Downloads ~33 GB in total, 6 shards at a time, deleting each after scanning.
# Output: results/training_data_check.jsonl, one line per shard.
BASE=https://olmo-data.org/preprocessed/tulu_flan/v2-decontaminated-60M-shots_all-upweight_1-dialog_false-sep_newline/train/gpt-neox-olmo-dolma-v1_5
TMP=${TMPDIR:-/tmp}/training_data_check; OUT=results/training_data_check.jsonl
mkdir -p $TMP && : > $OUT
seq -w 0 65 | xargs -P 6 -I{} bash -c "f=$TMP/part-{}.npy; curl -sS -f --retry 3 -o \$f $BASE/part-{}-00000.npy && python3 -I training_data_check.py \$f 2>/dev/null >> $OUT; rm -f \$f"
echo "scanned $(wc -l < $OUT) shards"
