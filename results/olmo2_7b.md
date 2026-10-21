# OLMo 2 extension: the same model before and after it saw the leaked copies

Pre-specified in `preregistration_olmo2.md`. Model: OLMo-2-1124-7B. Exposure:
6,701 of 9,334 benchmark sentences occur in the Dolmino FLAN files (23,746 copies) and 127 in the Tulu 3 SFT mixture (157 copies).

## P1. Sentence level: does more exposure mean a larger likelihood gain?

`gain ~ log1p(dose) + log1p(dose_pile) + length`, SEs clustered by item.

| step | scorer | beta log1p(dose) | 95% CI | p | n |
|---|---|---|---|---|---|
| pretraining → mid-training run 1 | logp_sum | +0.016 | [-0.038, +0.071] | 0.559 | 9,334 |
| pretraining → mid-training run 1 | logp_mean | +0.022 | [+0.007, +0.037] | 0.004 | 9,334 |
| pretraining → mid-training run 1 | full_logp_sum | +0.127 | [+0.063, +0.192] | <0.001 | 9,334 |
| pretraining → mid-training run 2 | logp_sum | +0.008 | [-0.049, +0.065] | 0.790 | 9,334 |
| pretraining → mid-training run 2 | logp_mean | +0.023 | [+0.008, +0.038] | 0.003 | 9,334 |
| pretraining → mid-training run 2 | full_logp_sum | +0.101 | [+0.034, +0.168] | 0.003 | 9,334 |
| pretraining → mid-training run 3 | logp_sum | +0.005 | [-0.049, +0.059] | 0.863 | 9,334 |
| pretraining → mid-training run 3 | logp_mean | +0.021 | [+0.006, +0.035] | 0.005 | 9,334 |
| pretraining → mid-training run 3 | full_logp_sum | +0.110 | [+0.044, +0.175] | <0.001 | 9,334 |
| pretraining → released base (merged) | logp_sum | +0.013 | [-0.041, +0.068] | 0.626 | 9,334 |
| pretraining → released base (merged) | logp_mean | +0.023 | [+0.008, +0.037] | 0.002 | 9,334 |
| pretraining → released base (merged) | full_logp_sum | +0.115 | [+0.051, +0.180] | <0.001 | 9,334 |
| base → SFT | logp_sum | -0.510 | [-1.329, +0.308] | 0.222 | 9,334 |
| base → SFT | logp_mean | +0.072 | [-0.031, +0.175] | 0.169 | 9,334 |
| base → SFT | full_logp_sum | -0.044 | [-0.915, +0.827] | 0.921 | 9,334 |

## P2. Bias score: does exposure move the within-pair comparison? (CrowS-Pairs)

`Δlogp(after) − Δlogp(before) ~ Δlog1p(dose) + Δlog1p(dose_pile) + Δlength`, HC1.
Scores count pairs where the stereotypical sentence is more likely; ties count as not.
The bound removes an exposure effect at either end of the 95% CI from the after-model's
untied pair differences.

| step | scorer | beta | 95% CI | p | CrowS before → after | ties after | score at CI ends |
|---|---|---|---|---|---|---|---|
| pretraining → mid-training run 1 | logp_sum | -0.056 | [-0.185, +0.074] | 0.401 | 65.45 → 64.26 | 100 | 66.91 to 62.47 |
| pretraining → mid-training run 1 | logp_mean | -0.009 | [-0.020, +0.002] | 0.116 | 65.45 → 64.26 | 100 | 67.31 to 63.26 |
| pretraining → mid-training run 1 | full_logp_sum | -0.023 | [-0.173, +0.126] | 0.760 | 66.91 → 65.85 | 0 | 66.31 to 65.25 |
| pretraining → mid-training run 2 | logp_sum | -0.040 | [-0.172, +0.092] | 0.555 | 65.45 → 65.05 | 100 | 67.24 to 62.40 |
| pretraining → mid-training run 2 | logp_mean | -0.006 | [-0.017, +0.005] | 0.288 | 65.45 → 65.05 | 100 | 67.44 to 62.86 |
| pretraining → mid-training run 2 | full_logp_sum | +0.010 | [-0.137, +0.158] | 0.892 | 66.91 → 66.38 | 0 | 67.11 to 65.58 |
| pretraining → mid-training run 3 | logp_sum | -0.033 | [-0.172, +0.106] | 0.643 | 65.45 → 64.59 | 100 | 66.98 to 62.20 |
| pretraining → mid-training run 3 | logp_mean | -0.007 | [-0.018, +0.003] | 0.182 | 65.45 → 64.59 | 100 | 67.24 to 63.00 |
| pretraining → mid-training run 3 | full_logp_sum | +0.019 | [-0.145, +0.183] | 0.821 | 66.91 → 66.58 | 0 | 67.24 to 65.58 |
| pretraining → released base (merged) | logp_sum | -0.051 | [-0.180, +0.077] | 0.434 | 65.45 → 64.26 | 100 | 67.51 to 62.80 |
| pretraining → released base (merged) | logp_mean | -0.008 | [-0.018, +0.003] | 0.139 | 65.45 → 64.26 | 100 | 67.84 to 63.20 |
| pretraining → released base (merged) | full_logp_sum | -0.003 | [-0.151, +0.145] | 0.968 | 66.91 → 66.11 | 0 | 66.91 to 65.19 |
| base → SFT | logp_sum | -0.331 | [-0.893, +0.231] | 0.248 | 64.26 → 61.80 | 100 | 61.94 to 61.80 |
| base → SFT | logp_mean | -0.025 | [-0.078, +0.027] | 0.342 | 64.26 → 61.80 | 100 | 61.94 to 61.74 |
| base → SFT | full_logp_sum | -0.123 | [-0.861, +0.615] | 0.744 | 66.11 → 66.31 | 0 | 66.45 to 66.25 |

## P3. The highest-dose sentences

Gain in log-likelihood from before to after, with the sentence's dose for that step.
For the StereoSet sentence, the percentile among all StereoSet gains is in brackets.

| step | scorer | `crows-0-more` | `crows-0-less` | `crows-50-more` | `crows-50-less` | `ss-62fa295b1bf1e1c85973db251baec1c0` |
|---|---|---|---|---|---|---|
| pretraining → mid-training run 1 | logp_sum | +9.01 (dose 926) | +5.88 (dose 1) | +1.68 (dose 448) | +1.68 (dose 1) | +0.32 (dose 1940) [70th] |
| pretraining → mid-training run 1 | full_logp_sum | +9.77 (dose 926) | +6.77 (dose 1) | +3.44 (dose 448) | +2.18 (dose 1) | +2.14 (dose 1940) [94th] |
| pretraining → mid-training run 2 | logp_sum | +7.72 (dose 926) | +4.30 (dose 1) | +1.63 (dose 448) | +1.63 (dose 1) | +0.82 (dose 1940) [84th] |
| pretraining → mid-training run 2 | full_logp_sum | +7.98 (dose 926) | +4.51 (dose 1) | +2.78 (dose 448) | +1.70 (dose 1) | +3.38 (dose 1940) [98th] |
| pretraining → mid-training run 3 | logp_sum | +9.87 (dose 926) | +5.75 (dose 1) | +1.44 (dose 448) | +1.44 (dose 1) | +0.65 (dose 1940) [80th] |
| pretraining → mid-training run 3 | full_logp_sum | +11.43 (dose 926) | +6.63 (dose 1) | +2.42 (dose 448) | +1.31 (dose 1) | +2.97 (dose 1940) [97th] |
| pretraining → released base (merged) | logp_sum | +8.47 (dose 926) | +5.18 (dose 1) | +1.63 (dose 448) | +1.63 (dose 1) | +0.54 (dose 1940) [78th] |
| pretraining → released base (merged) | full_logp_sum | +9.35 (dose 926) | +5.87 (dose 1) | +2.93 (dose 448) | +1.69 (dose 1) | +2.71 (dose 1940) [96th] |
| base → SFT | logp_sum | -18.18 (dose 11) | -15.91 (dose 0) | -4.40 (dose 3) | -4.40 (dose 0) | -0.45 (dose 14) [65th] |
| base → SFT | full_logp_sum | -18.10 (dose 11) | -16.33 (dose 0) | -4.91 (dose 3) | -6.08 (dose 0) | -4.53 (dose 14) [46th] |

## Exposed vs unexposed sentences within StereoSet instances (mid-training steps)

Instance fixed effects, role, length and Pile dose as controls; CR1 on instance.

| step | scorer | beta exposed | 95% CI | p | n sentences / instances |
|---|---|---|---|---|---|
| pretraining → mid-training run 1 | logp_sum | +0.036 | [-0.032, +0.104] | 0.296 | 3,147 / 1,049 |
| pretraining → mid-training run 1 | logp_mean | +0.029 | [-0.005, +0.063] | 0.099 | 3,147 / 1,049 |
| pretraining → mid-training run 2 | logp_sum | +0.030 | [-0.040, +0.099] | 0.403 | 3,147 / 1,049 |
| pretraining → mid-training run 2 | logp_mean | +0.023 | [-0.010, +0.057] | 0.175 | 3,147 / 1,049 |
| pretraining → mid-training run 3 | logp_sum | +0.017 | [-0.049, +0.084] | 0.611 | 3,147 / 1,049 |
| pretraining → mid-training run 3 | logp_mean | +0.026 | [-0.007, +0.058] | 0.120 | 3,147 / 1,049 |
| pretraining → released base (merged) | logp_sum | +0.026 | [-0.040, +0.092] | 0.437 | 3,147 / 1,049 |
| pretraining → released base (merged) | logp_mean | +0.025 | [-0.008, +0.058] | 0.137 | 3,147 / 1,049 |

## Deviations from preregistration_olmo2.md

- 20 texts are each shared by two benchmark sentences (duplicate CrowS pairs and StereoSet
  items). Both sentences get the text's count, as with infini-gram. This was decided while
  building `data/dose_olmo2.csv`, before this analysis was first run.
- Scored in fp16, not fp32, because fp32 does not fit on the one idle 24 GB GPU. The
  before and after checkpoints use the same dtype.
- The SFT step uses `dose_tulu3_1124`, the counts in `tulu-3-sft-olmo-2-mixture`, which
  is the mixture this model's SFT used. The plan names the 0225 mixture, which the 1B used.
- The predictions are the plan's, unchanged after the 1B results. This setup was
  committed and pushed before any 7B checkpoint was scored.
