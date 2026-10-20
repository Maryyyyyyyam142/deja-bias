# Task 2, Phase A — Gate Verdict

> **Corrections added 2026-10-19 — read §8 first.** The §1 decomposition table and
> its headline are withdrawn, H5 does not survive few-cluster standard errors, and
> "pre-registered" here means specified in `preregistration.md`, which was never
> posted to a registry. §1–§7 are otherwise left as written in August.

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

## 8. Corrections, 2026-10-19

Added after re-checking this file against the data. Read §1–§7 with these
corrections. The new checks are in `robustness.py`, with output in
`results/robustness.md`; none of them is in the pre-specified plan.

1. **"Pre-registered" means pre-specified, not registered.** `preregistration.md`
   was written before any model was scored but was never posted to a registry. In
   the paper, call it an analysis plan written before scoring, and say it was not
   publicly registered. The only evidence of the order is the file timestamps: plan
   12:07, scoring job 12:16 on 2026-08-15. They are recorded in
   `archive/MANIFEST_phaseA_2026-10-19.txt` and in the repository's first commit.
2. **The §1 decomposition and its headline are withdrawn.** The table treats
   `dose_dolma` as general v1.7 exposure and `dose_natinst` as the injected copies.
   In fact 96.7% of sentences in the NatInst files are in Dolma v1.7, against 2.0% of
   the rest, and the two log-doses correlate at 0.79. `dose_dolma` is mostly the
   injected route, counted in the corpus. The negative `dose_natinst` coefficient
   with `dose_dolma` held fixed cannot show that "the contamination route does not
   move likelihood".
   - §4's claim that the sentence-level result "excludes +0.2" falls with it.
   - The main test is now the pair-level DiD (H3), and its 95% CI still includes +0.2.
   - The code behind the table had not been saved; it is now `robustness.py` §A.
   - H1–H3, the §3 benchmark scores and H8 do not depend on the table.
3. **`dose_natinst` was double-counted.** `build_variables.py` counted each
   single-sentence NatInst instance twice: it lacked the `len(parts) > 1` guard in
   `ni_provenance.py`. This is now fixed and `data/variables.csv` has been regenerated.
   - 6,534 values halved; nothing else changed.
   - No sentence moved between zero and non-zero, so `analysis.py` output is
     identical before and after the fix.
   - CrowS pairs are 2 copies to 1 for 1,496 of 1,508 pairs. The pre-registration's
     §1.2 figure, "+2 for 1,404 of 1,498", reflects the double count.
   - The five NatInst task files (tasks 277, 279, 280, 316, 317) had been read from
     `/tmp` and lost. They have been re-downloaded into `data/ni_tasks/`, and they
     reproduce `data/ni_copies.json` exactly.
4. **Standard errors.** The planned SEs cluster on bias_type (9 CrowS clusters) with
   a normal reference, which over-rejects with this few clusters. `robustness.md` §C
   adds four alternatives: t(G−1), a wild cluster bootstrap, pair-robust SEs and
   writer clusters.
   - **H1–H3 stay null** under every choice (all p > 0.27).
   - **H5 does not survive.** The top-decile slopes were p = 0.007 (v1.7) and
     p < 0.001 (Pythia); under the wild bootstrap they are 0.11 and 0.14.
     Figure 2 shows the planned SEs.
   - §2's "H5 is the methodological result worth publishing on its own" and §7
     option 2 are withdrawn.
   - What remains of H5 is narrower: the top-decile slope has about the same size in
     models that never saw the copies (+0.50 / +0.38 / +0.43).
5. **CrowS ties.** Under both planned scorers, 102 of 1,508 pairs are exact ties.
   Nothing is scored after the swapped word, and the scored tokens before it share a
   left context, so both members get identical log-probabilities in any causal LM.
   The CrowS score counts ties as non-stereotypical. Excluding them moves OLMo v1.7
   from 62.86 to 67.43, so report the tie count. The full-sentence scorer has no ties.
6. **State the null as a bound.** Removing an exposure effect at either end of the
   H3 95% CI gives OLMo v1.7 a CrowS score between 59.4 and 65.2 under scorer A
   (59.8–66.5 under B), against a planned score of 62.86 (`robustness.md` §D4). The
   data cannot rule out a shift of about 3.5 points either way. The supported claim
   is "no detectable effect, bounded at about ±3.5 points", not "no effect".
7. **Exploratory checks** (`robustness.md` §D1–D3).
   - The pair-level null holds under the full-sentence scorer.
   - The 836- and 397-copy sentences gain no more than their 1-copy partners.
   - Within StereoSet instances, being in the leaked files goes with a gain of
     +0.08 nats (p = 0.10). That is at most a very small effect on sentence
     likelihood.
8. **Treatment delivery: confirmed (2026-10-20).** `training_data_check.sh` scanned
   all 66 `tulu_flan` shards of OLMo-1.7's stage-1 training data (16.54B tokens). The
   shard list comes from allenai/OLMo `configs/official-0724/OLMo-7B-0424.yaml`.
   - The training files hold exactly the copies the Dolma v1.7 index reports: 836,
     1,728 and 397 for the three highest-dose exemplars, and 1 for the 836-copy
     sentence's partner (`results/training_data_check.jsonl`).
   - One pass over the stage-1 mix is 1,715B tokens, and stage 1 ran for 2T. So every
     copy was trained on at least once, about 1.17 times on average.
   - Per the model card, stage 2 trained on "all available ... Flan data" for the
     final 50B tokens while the learning rate decayed to zero.
   - The null is therefore not a delivery failure. Sentences trained on roughly
     1,000–2,000 times, as instruction-formatted exemplars, show no detectable gain
     in bare-sentence likelihood.

## 9. OLMo 2 extension, 2026-10-20

The extension was pre-specified in `preregistration_olmo2.md`, which was pushed to
GitHub as commit `9156040` before any OLMo 2 model was scored. The output is in
`results/olmo2.md`.

**Design.** OLMo-2-0425-1B at the end of pretraining is compared with:

- the same model after each of three independent mid-training runs on
  Dolmino-mix-1124;
- the merged release;
- the release after SFT on Tulu 3.

OLMo 2's pretraining data contains none of the copies. Dolmino's FLAN part holds
23,746 copies of 6,701 benchmark sentences (`data/dose_olmo2.csv`), and Tulu 3 holds
158 copies of 127 sentences. So this compares the same weights before and after
exposure, which is cleaner than Phase A's OLMo-1.0 vs OLMo-1.7.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | Sentence-level β null or small positive (< 0.2), with the same sign in all runs | Scorer A: −0.018 to +0.010, all p > 0.5. Scorer B: +0.008 to +0.014, p 0.05–0.25. Full-sentence scorer: +0.068 to +0.093 in each run (p 0.007–0.05), and +0.004 for the merged model | **Confirmed**: at most a small effect, well under 0.2 |
| P2 | Bias-score DiD β ≈ 0 in all runs | All 12 run × scorer CIs cover 0 (p 0.40–0.95); the merged and SFT models are null too. The CrowS score is bounded within about ±2–3 points | **Confirmed** |
| P3 | The top exemplars gain no more than their partners and stay below the 95th percentile | The 926-copy CrowS sentence gains 1.3–2.5 nats more than its 1-copy partner in every run, and 4.1 more in the merged model. The 1,940-copy StereoSet sentence sits at the 97th–99th percentile (full scorer) in every run, but the 30th in the merged model. The 448-copy sentence shows no excess | **Refuted for the two most-copied sentences**: modest memorisation within individual runs |
| P4 | No dose effect at SFT | P1 and P2 are null for base → SFT | **Confirmed** |

**Reading.**

- **Bias score.** In the cleanest design available, exposure during mid-training
  leaves the bias score unmoved, and this replicates across three independent runs.
- **The most-copied sentences.** The two most-copied exemplars show some
  memorisation, but the pair metric doesn't register it. Whole files leak together,
  and only a handful of sentences are copied hundreds of times.

**Limitations.**

- Only the 1B model has been tested.
- The doses count copies in the Dolmino data. How many copies each 50B-token run
  actually saw is not recorded.
- P3 rests on two sentences.

**Next.** Run OLMo 2 7B, which fits on one idle GPU, under the same plan unchanged.
