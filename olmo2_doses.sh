#!/bin/bash
# Build data/dose_olmo2.csv: scan all 209 Dolmino-mix-1124 FLAN files (~33 GB, 8 at a
# time, each deleted after scanning) and the Tulu 3 SFT mixture used for OLMo-2-0425-1B-SFT.
# PY must point to a Python with pyahocorasick and pyarrow installed.
PY=${PY:-python3}; TMP=${TMPDIR:-/tmp}/olmo2_doses; mkdir -p $TMP/tulu3
HF=https://huggingface.co/datasets/allenai
: > $TMP/dolmino_counts.jsonl
seq -f "%04g" 0 208 | xargs -P 8 -I{} bash -c "f=$TMP/tulu_flan-{}.json.gz; \
  curl -sS -L -f --retry 3 -o \$f $HF/dolmino-mix-1124/resolve/main/data/flan/tulu_flan-{}.json.gz \
  && $PY -I olmo2_doses.py dolmino \$f >> $TMP/dolmino_counts.jsonl; rm -f \$f"
for i in 0 1 2 3 4 5; do
  curl -sS -L -f -o $TMP/tulu3/part$i.parquet \
    $HF/tulu-3-sft-olmo-2-mixture-0225/resolve/main/data/train-0000$i-of-00006.parquet
done
$PY -I olmo2_doses.py tulu3 $TMP/tulu3 > $TMP/tulu3_counts.json
$PY -I olmo2_doses.py build $TMP/dolmino_counts.jsonl $TMP/tulu3_counts.json
