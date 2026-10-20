# Analysis plan: OLMo 2 extension

Written 2026-10-20, before any OLMo 2 model was scored. It is committed in git before
`results/likelihoods_olmo2.csv` exists. The analysis it specifies is `olmo2.py`, which
is committed with it.

## Why OLMo 2

The Phase A comparison, OLMo-7B (Dolma v1.5) vs OLMo-1.7-7B (Dolma v1.7), changes the
whole corpus at once. OLMo 2 allows a cleaner design: **the same model, before and
after the training stage that adds the leaked copies.**

These exposure facts were established from the training data before any scoring:

- **Pretraining (OLMo-mix-1124):** no copies of the probe sentences (infini-gram:
  0 for the 836-, 1,728- and 397-copy sentences and for the task316 template).
- **Mid-training (Dolmino-mix-1124):** the FLAN part carries the copies (a scan of
  its 209 files is in progress).
- **SFT (Tulu 3, OLMo 2 mixtures):** a few copies, 156 in total, of 126 benchmark
  sentences, all from the FLAN v2 subset.

## Models

All checkpoints are `allenai/OLMo-2-0425-1B` unless stated otherwise:

- `stage1-step1907359-tokens4001B`: end of pretraining (before).
- `stage2-ingredient{1,2,3}-step23852-tokens51B`: three independent mid-training
  runs on Dolmino (after). They are three replications of the treatment.
- `main`: the released base model, a merge of the three runs.
- `allenai/OLMo-2-0425-1B-SFT`: instruction-tuned on Tulu 3.

Every checkpoint is scored with `scorer.py`: fp32, on one idle GPU, and with the
same scorers as Phase A (`logp_sum` and `logp_mean`), plus `full_logp_sum`.

## Exposure variables

- `dose_dolmino`: per-sentence count in the Dolmino FLAN files.
- `dose_tulu3`: per-sentence count in `tulu-3-sft-olmo-2-mixture-0225`.

Both are exact string matches over the data files, with overlapping matches
counted (`data/dose_olmo2.csv`).

## Tests and predictions

These are written before scoring. The steps are stage1 → each mid-training run,
stage1 → merged, and base → SFT.

| # | Test | Prediction |
|---|---|---|
| P1 | Sentence level: `gain ~ log1p(dose) + log1p(dose_pile) + length`, gain = logp(after) − logp(before) | β null or small positive (< 0.2 nats per log-unit, sum scorer), with the same sign in all three runs |
| P2 | Bias score: CrowS pair-level DiD, `Δlogp(after) − Δlogp(before) ~ Δlog1p(dose) + Δlog1p(dose_pile) + Δlength` | β ≈ 0, with a 95% CI that covers 0, in all three runs |
| P3 | Exemplars: the before → after gain of the 836-, 397- and 1,728-copy sentences, compared with their 1-copy partners and the StereoSet gain distribution | No more gain than their partners; below the 95th percentile |
| P4 | SFT step | No detectable dose effect, because the SFT dose is 0 for 99% of sentences |

The prediction rests on Phase A, which found no exemplar memorisation at 7B, and a
1B model memorises less.

## Inference

- **Sentence-level SEs:** clustered by item, meaning the CrowS-Pairs pair or the
  StereoSet instance.
- **Pair-level SEs:** HC1, with one row per pair. There is no few-cluster structure,
  so no bootstrap is needed.
- **Replication:** a result counts as replicated only if it holds in all three
  mid-training runs.
- **The null as a bound:** the null is also stated as a bound on the CrowS score, as
  in `robustness.md` §D4. Ties are held fixed.

## What would change the conclusion

- **If P2 is positive in all three runs,** exposure late in training *does* move the
  bias score. That becomes the main OLMo 2 result, and it qualifies the Phase A null.
- **If P1 is positive but P2 is null,** exposure raises sentence likelihood without
  moving the pair comparison that the bias score uses.
- **Changes after scoring:** any change to this plan made after scoring is listed as
  a deviation in `results/olmo2.md`.
