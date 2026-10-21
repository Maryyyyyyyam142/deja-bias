# OLMo 2 extension: the same model before and after it saw the leaked copies

Pre-specified in `preregistration_olmo2.md`. Model: OLMo-2-0425-1B. Exposure:
6,701 of 9,334 benchmark sentences occur in the Dolmino FLAN files (23,746 copies) and 127 in the Tulu 3 SFT mixture (158 copies).

## P1. Sentence level: does more exposure mean a larger likelihood gain?

`gain ~ log1p(dose) + log1p(dose_pile) + length`, SEs clustered by item.

| step | scorer | beta log1p(dose) | 95% CI | p | n |
|---|---|---|---|---|---|
| pretraining → mid-training run 1 | logp_sum | -0.003 | [-0.056, +0.051] | 0.924 | 9,334 |
| pretraining → mid-training run 1 | logp_mean | +0.013 | [-0.001, +0.028] | 0.070 | 9,334 |
| pretraining → mid-training run 1 | full_logp_sum | +0.073 | [+0.006, +0.139] | 0.032 | 9,334 |
| pretraining → mid-training run 2 | logp_sum | -0.018 | [-0.073, +0.037] | 0.516 | 9,334 |
| pretraining → mid-training run 2 | logp_mean | +0.008 | [-0.006, +0.023] | 0.250 | 9,334 |
| pretraining → mid-training run 2 | full_logp_sum | +0.068 | [+0.000, +0.135] | 0.050 | 9,334 |
| pretraining → mid-training run 3 | logp_sum | +0.010 | [-0.044, +0.065] | 0.706 | 9,334 |
| pretraining → mid-training run 3 | logp_mean | +0.014 | [-0.000, +0.029] | 0.054 | 9,334 |
| pretraining → mid-training run 3 | full_logp_sum | +0.093 | [+0.025, +0.161] | 0.007 | 9,334 |
| pretraining → released base (merged) | logp_sum | -0.012 | [-0.086, +0.063] | 0.760 | 9,334 |
| pretraining → released base (merged) | logp_mean | +0.020 | [-0.000, +0.040] | 0.055 | 9,334 |
| pretraining → released base (merged) | full_logp_sum | +0.004 | [-0.091, +0.099] | 0.940 | 9,334 |
| base → SFT | logp_sum | -0.482 | [-1.423, +0.460] | 0.316 | 9,334 |
| base → SFT | logp_mean | +0.077 | [-0.102, +0.256] | 0.400 | 9,334 |
| base → SFT | full_logp_sum | -0.308 | [-1.396, +0.780] | 0.579 | 9,334 |

## P2. Bias score: does exposure move the within-pair comparison? (CrowS-Pairs)

`Δlogp(after) − Δlogp(before) ~ Δlog1p(dose) + Δlog1p(dose_pile) + Δlength`, HC1.
Scores count pairs where the stereotypical sentence is more likely; ties count as not.
The bound removes an exposure effect at either end of the 95% CI from the after-model's
untied pair differences.

| step | scorer | beta | 95% CI | p | CrowS before → after | ties after | score at CI ends |
|---|---|---|---|---|---|---|---|
| pretraining → mid-training run 1 | logp_sum | -0.033 | [-0.147, +0.082] | 0.576 | 62.33 → 61.14 | 114 | 63.40 to 59.55 |
| pretraining → mid-training run 1 | logp_mean | -0.004 | [-0.013, +0.005] | 0.401 | 61.94 → 60.81 | 124 | 63.20 to 59.48 |
| pretraining → mid-training run 1 | full_logp_sum | -0.014 | [-0.164, +0.137] | 0.858 | 62.60 → 63.66 | 0 | 64.85 to 62.27 |
| pretraining → mid-training run 2 | logp_sum | -0.016 | [-0.121, +0.090] | 0.772 | 62.33 → 62.53 | 113 | 64.06 to 60.94 |
| pretraining → mid-training run 2 | logp_mean | -0.001 | [-0.010, +0.009] | 0.883 | 61.94 → 62.47 | 123 | 63.66 to 60.61 |
| pretraining → mid-training run 2 | full_logp_sum | +0.013 | [-0.124, +0.149] | 0.854 | 62.60 → 63.73 | 0 | 64.79 to 62.40 |
| pretraining → mid-training run 3 | logp_sum | -0.029 | [-0.135, +0.077] | 0.590 | 62.33 → 61.27 | 112 | 63.59 to 59.62 |
| pretraining → mid-training run 3 | logp_mean | -0.003 | [-0.012, +0.006] | 0.545 | 61.94 → 60.74 | 128 | 62.47 to 59.55 |
| pretraining → mid-training run 3 | full_logp_sum | -0.004 | [-0.148, +0.139] | 0.953 | 62.60 → 63.06 | 0 | 64.26 to 62.53 |
| pretraining → released base (merged) | logp_sum | -0.041 | [-0.209, +0.126] | 0.630 | 62.33 → 60.21 | 114 | 63.53 to 58.49 |
| pretraining → released base (merged) | logp_mean | -0.005 | [-0.019, +0.008] | 0.428 | 61.94 → 59.95 | 123 | 63.20 to 58.69 |
| pretraining → released base (merged) | full_logp_sum | +0.024 | [-0.182, +0.230] | 0.822 | 62.60 → 61.87 | 0 | 62.20 to 60.08 |
| base → SFT | logp_sum | +0.143 | [-0.862, +1.148] | 0.781 | 60.21 → 57.82 | 114 | 58.16 to 57.82 |
| base → SFT | logp_mean | +0.025 | [-0.066, +0.116] | 0.587 | 59.95 → 57.69 | 118 | 57.96 to 57.56 |
| base → SFT | full_logp_sum | +0.125 | [-0.972, +1.222] | 0.823 | 61.87 → 60.94 | 0 | 60.94 to 60.68 |

## P3. The highest-dose sentences

Gain in log-likelihood from before to after, with the sentence's dose for that step.
For the StereoSet sentence, the percentile among all StereoSet gains is in brackets.

| step | scorer | `crows-0-more` | `crows-0-less` | `crows-50-more` | `crows-50-less` | `ss-62fa295b1bf1e1c85973db251baec1c0` |
|---|---|---|---|---|---|---|
| pretraining → mid-training run 1 | logp_sum | +6.73 (dose 926) | +4.27 (dose 1) | +1.96 (dose 448) | +1.96 (dose 1) | +1.09 (dose 1940) [88th] |
| pretraining → mid-training run 1 | full_logp_sum | +7.35 (dose 926) | +4.90 (dose 1) | +3.56 (dose 448) | +3.91 (dose 1) | +2.98 (dose 1940) [97th] |
| pretraining → mid-training run 2 | logp_sum | +9.11 (dose 926) | +7.77 (dose 1) | +2.39 (dose 448) | +2.39 (dose 1) | +1.18 (dose 1940) [90th] |
| pretraining → mid-training run 2 | full_logp_sum | +9.25 (dose 926) | +7.97 (dose 1) | +3.90 (dose 448) | +4.42 (dose 1) | +3.47 (dose 1940) [99th] |
| pretraining → mid-training run 3 | logp_sum | +7.93 (dose 926) | +6.15 (dose 1) | +2.69 (dose 448) | +2.69 (dose 1) | +1.48 (dose 1940) [93th] |
| pretraining → mid-training run 3 | full_logp_sum | +8.59 (dose 926) | +6.75 (dose 1) | +3.95 (dose 448) | +4.63 (dose 1) | +3.26 (dose 1940) [98th] |
| pretraining → released base (merged) | logp_sum | +9.72 (dose 926) | +5.66 (dose 1) | +3.86 (dose 448) | +3.86 (dose 1) | +0.62 (dose 1940) [73th] |
| pretraining → released base (merged) | full_logp_sum | +13.16 (dose 926) | +8.34 (dose 1) | +6.70 (dose 448) | +6.91 (dose 1) | -1.52 (dose 1940) [30th] |
| base → SFT | logp_sum | -15.10 (dose 11) | -12.54 (dose 0) | -11.64 (dose 3) | -11.64 (dose 0) | -0.52 (dose 14) [54th] |
| base → SFT | full_logp_sum | -19.26 (dose 11) | -15.61 (dose 0) | -12.93 (dose 3) | -12.99 (dose 0) | -2.39 (dose 14) [74th] |

## Exposed vs unexposed sentences within StereoSet instances (mid-training steps)

Instance fixed effects, role, length and Pile dose as controls; CR1 on instance.

| step | scorer | beta exposed | 95% CI | p | n sentences / instances |
|---|---|---|---|---|---|
| pretraining → mid-training run 1 | logp_sum | -0.004 | [-0.075, +0.067] | 0.918 | 3,147 / 1,049 |
| pretraining → mid-training run 1 | logp_mean | +0.007 | [-0.027, +0.042] | 0.683 | 3,147 / 1,049 |
| pretraining → mid-training run 2 | logp_sum | -0.000 | [-0.072, +0.072] | 0.998 | 3,147 / 1,049 |
| pretraining → mid-training run 2 | logp_mean | +0.007 | [-0.027, +0.041] | 0.673 | 3,147 / 1,049 |
| pretraining → mid-training run 3 | logp_sum | +0.012 | [-0.062, +0.085] | 0.756 | 3,147 / 1,049 |
| pretraining → mid-training run 3 | logp_mean | +0.009 | [-0.026, +0.043] | 0.616 | 3,147 / 1,049 |
| pretraining → released base (merged) | logp_sum | -0.071 | [-0.183, +0.042] | 0.219 | 3,147 / 1,049 |
| pretraining → released base (merged) | logp_mean | -0.009 | [-0.064, +0.045] | 0.742 | 3,147 / 1,049 |

## Deviations from preregistration_olmo2.md

- 20 texts are each shared by two benchmark sentences (duplicate CrowS pairs and StereoSet
  items). Both sentences get the text's count, as with infini-gram. This was decided while
  building `data/dose_olmo2.csv`, before this analysis was first run.
- The SFT checkpoint ships `pytorch_model.bin`, not safetensors, so it was scored after a
  second download. No analysis was changed after scoring.
- After the 1B results were seen, `olmo2.py` was edited only in two ways: this deviations
  text was added, and on 2026-10-21 a `--size` option was added for the 7B run. The 1B
  numbers are unchanged, which was verified by regenerating this file.
