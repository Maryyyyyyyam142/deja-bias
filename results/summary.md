# Task 2, Phase A — Gate Verdict

**Verdict: STOP AND REPORT (§6, row 2).** β is null under both scoring variants,
and negative for the contamination-specific estimand. This is a well-identified
null on a documented contamination route.

Panel: OLMo-7B v1.5 (Dolma v1.5-sample, clean) · OLMo-7B v1.7 (Dolma v1.7,
contaminated) · Pythia-6.9B (Pile, clean). 9,334 sentences × 3 models × 2 scorers
= 28,002 likelihoods. Dose coverage 9,334/9,334 on both indices (19,529 cached
infini-gram responses).

---

## 1. The headline result

**Corpus exposure does move sentence likelihood. The contamination route does
not, and the bias score does not move at all.**

Decomposing v1.7 exposure into the injected component and everything else
(outcome = same sentence's log-likelihood under v1.7 minus v1.5, same tokenizer
and architecture, so sentence identity differences out exactly; n=9,334, SEs
clustered by `bias_type`):

| Predictor | β, sum log-prob | 95% CI | p | β, mean log-prob | p |
|---|---|---|---|---|---|
| `log(1+dose_natinst)` — **the injected copies** | **−0.193** | [−0.410, +0.024] | 0.082 | −0.017 | 0.21 |
| `log(1+dose_dolma)` — general v1.7 exposure | +0.275 | [+0.059, +0.490] | 0.012 | +0.040 | <0.001 |
| `log(1+dose_pile)` — natural web frequency | −0.425 | [−0.625, −0.225] | <0.001 | −0.075 | <0.001 |

General exposure behaves as the literature predicts. The **contamination-specific
coefficient is negative**, and its CI excludes the pre-registered +0.2. Read
without the decomposition, `dose_dolma` alone gives +0.141 (p=0.0055) and looks
like a confirmation — that is the trap this design was built to catch.

**Why the bias score is unmoved:** CrowS-Pairs scores a *within-pair* comparison,
and the leakage from Task 1 is whole-file — both members of a pair arrive
together. Δdose is near-constant (+2 NatInst copies for 1,404 of 1,498 pairs), so
whatever exposure does, it does to both members, and it cancels in the difference.
The metric is protected by its own differencing structure.

## 2. Every pre-registered prediction, marked

| # | Prediction | Result | Verdict |
|---|---|---|---|
| H1 | β>0 in contaminated arm, +0.2 | −0.037 (p=0.77) sum; −0.003 (p=0.84) mean | **REFUTED** |
| H2 | β≈0 in clean arms, \|β\|<0.10 | −0.130 (v1.5), −0.081 (Pythia), both null | **CONFIRMED** |
| H3 | β_DiD>0, +0.2 | pair-level +0.076 (p=0.54); sentence-level contamination-specific −0.193 (p=0.08) | **REFUTED** |
| H4 | sign agreement >50%, 55% | 58.3% contaminated vs 59.7% clean v1.5, 56.7% Pythia | **REFUTED as evidence** — no differential; the level is the bias signal, not dose |
| H5 | effect tail-driven, ≥2× in top decile | top-decile β: v1.7 +0.496 (p=0.007), **v1.5 +0.378, Pythia +0.425 (p<0.0001)** | **REFUTED** — largest in a clean model |
| H6 | StereoSet full−zero gap, +0.8 | v1.7 ss gap +5.60 (sum) but −1.02 (mean); CI [54.9, 66.0] nearly covers full | **REFUTED** — not robust, flips sign |
| H7 | ICAT distorted for high-dose `unrelated` | ICAT gaps −9.73 (sum) / +3.38 (mean), signs flip | **REFUTED** (Phase C scope; reported for completeness) |
| H8 | exemplars show elevated logp | 836-copy sentence gained **+7.32**; its 1-copy partner **+8.17** | **REFUTED** |

**H5 is the methodological result worth publishing on its own.** A naive
dose-response on the contaminated model alone returns β=+0.496, p=0.007 — clean
significance in the predicted direction. The identical analysis on Pythia, which
never saw an injected copy, returns +0.425, p<0.0001. The "effect" is natural
sentence frequency. Without the clean-arm placebo we would have reported it.

**H8 kills the §5.4 case study.** The 836:1 sentence is not detectably memorised:
v1.7 is broadly stronger than v1.5 on everything, and that baseline shift
(+7 to +8 nats on long sentences) swamps dose. The figure that was meant to
anchor the paper shows the opposite of its intended claim.

## 3. Benchmark scores

| Model | CrowS | StereoSet ss | lm | ICAT |
|---|---|---|---|---|
| OLMo-7B v1.5 (clean) | 64.52 | 66.10 | 89.70 | 60.82 |
| OLMo-7B v1.7 (contaminated) | 62.86 | 66.05 | 90.22 | 61.26 |
| Pythia-6.9B (clean) | 61.14 | 61.97 | 84.05 | 63.93 |

The contaminated model scores **lower** on CrowS than the clean v1.5. Confounded
by everything else that changed between corpus versions, so it carries no weight —
but it points away from "contamination inflates apparent bias."

## 4. Power — where the null is strong and where it is not

- **Sentence-level decomposition (n=9,334): well powered.** The `dose_natinst` CI
  [−0.410, +0.024] excludes +0.2. The pre-registered effect is ruled out.
- **Pair-level test (n=1,508): underpowered.** Clustered SE 0.122 gives a minimum
  detectable effect of ~0.34 nats at 80% power, so it could never have detected
  +0.2. The pair-level null is consistent with, but not evidence for, no effect.
  Do not cite it as the null; cite the sentence-level result.
- **§5.2 for CrowS is not computable at all** — 0 of 1,508 pairs have both members
  at zero dose. StereoSet's zero-dose subset is n=268 instances, and its bootstrap
  CIs are wide enough to cover the full-benchmark score.

## 5. Deviations from the pre-registration

1. **Added the sentence-level DiD** (same sentence, v1.7 − v1.5). Pre-registered
   plan specified pair-level only. Added because the pair-level treatment is
   near-constant; this is a cleaner estimator of the *same* pre-registered
   estimand (H3), not a new hypothesis. Both are reported.
2. **Added the `dose_natinst` / `dose_dolma` decomposition.** Not pre-specified.
   Without it, `dose_dolma` alone is confounded by general v1.7 exposure and
   returns a false positive. Reported as the primary contamination test.
3. **H7 (ICAT) computed in Phase A** rather than deferred to Phase C, since the
   scores were already in hand. Scope unchanged otherwise.
4. Palette validator (`node`) unavailable; figures use the dataviz reference
   palette's categorical slots 1–3 unmodified rather than a re-validated set.

## 6. Prior-work flag, carried forward from `preregistration.md`

**Reif & Schwartz (2024)** exists (arXiv:2405.02743, NAACL 2024) but I could not
verify either claim the brief attributes to it — the ~20% FLAN/SuperNI figure or
the contamination remark. Do not cite it for those until someone reads the full
text. Novelty claim 3 (sentence-level dose-response) must be narrowed: the general
duplication→memorisation relationship is established, and arXiv:2503.22362 already
does frequency→behaviour on Dolma/OLMo with a minimal-pair design. Claims 1 and 2
are intact.

## 7. What to do next

The §1 claim — "contamination perturbs the measured bias score in a direction set
by an arbitrary authoring decision" — is **not supported**. Exposure predicts
likelihood; the injected copies do not; and the bias metric is insulated because
it differences two sentences that were leaked together.

Three honest options, in order of my preference:

1. **Write the null.** It is well-identified, pre-registered, and useful: bias
   benchmarks are contaminated (Task 1: 65.7% of probed sentences present, whole
   files via SuperNI→FLAN→Tulu→Dolma) and the contamination does not move the
   score. Pair it with the H5 placebo result as a methods contribution — naive
   dose-response analysis on contaminated corpora manufactures significance.
2. **Repoint at the placebo finding as the paper.** H5 generalises past bias
   benchmarks and is a reusable warning about contamination methodology.
3. **Do not run Phases B and C as specified.** They extend a design whose primary
   estimand has already returned a null; scaling the model panel will not change
   the sign of `dose_natinst`.
