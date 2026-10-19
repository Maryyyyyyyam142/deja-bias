# Pre-registration — Task 2: Does Corpus Exposure Move the Bias Score?

**Committed 2026-08-15, before any model was loaded or any likelihood computed.**
Torch/CUDA was installed and dose collection was running at the time of writing;
no scoring code had been executed. This file is not to be edited after commit.
Corrections go in `summary.md` marked as departures.

---

## 0. Literature check (§2), run before writing predictions

| Work | Status | Bearing on us |
|---|---|---|
| Longpre et al. 2023, *The Flan Collection* — arXiv:2301.13688 | **Verified** | Documents the route. Cite as the route's own description; not our discovery. |
| Sainz et al. 2023, *NLP Evaluation in trouble* — arXiv:2310.18018, Findings EMNLP 2023 | **Verified** | General framing. |
| Reif & Schwartz 2024, *Beyond Performance* — arXiv:2405.02743, NAACL 2024 | **Exists; attributed claims NOT verified** | See below. |

**Flag on Reif & Schwartz.** The paper exists and does build its evaluation suite
from Super-NaturalInstructions across 279 tasks and ten LLMs. But I could not
verify the two specific claims the task brief attributes to it — that ~20% of FLAN
consists of SuperNI examples, and that evaluation of Flan-tuned models is likely
contamination-affected. Neither appears in the abstract or the ACL Anthology
landing page, and the PDF did not extract. **Do not cite it for those claims until
someone reads the full text.** If the claims are not in it, our §2 differentiation
paragraph needs rewriting, and the closest prior work is something else.

**Novelty claims, re-assessed:**

1. *Applying this to social-bias benchmarks* — **intact.** No work found that
   audits CrowS-Pairs/StereoSet contamination and its metric consequences.
2. *Direction and magnitude of the effect on bias metrics* — **intact.**
3. *Dose-response at the individual sentence level* — **weakened, must be
   narrowed.** The general log-linear duplication→memorization relationship is
   established (Carlini et al.; Kandpal et al. 2022; Lee et al. 2022), and
   frequency→behaviour on Dolma/OLMo specifically has been done at the entity
   level, including with a minimal-pair design (*Supposedly Equivalent Facts That
   Aren't?*, arXiv:2503.22362, which uses Wikidata inverse triples + Dolma counts
   + OLMo). We must claim only the bias-benchmark-specific instance and the
   metric consequence, and must cite 2503.22362 as methodologically closest.

Also checked: a recent audit reported "no evidence of contamination" for eval
material in C4/Pile/RedPajama/Dolma/DCLM via infini-gram. It is not a conflict —
it sampled 3 random 5–10-token substrings per case, a protocol that would not
detect what Task 1 found, and it did not target bias benchmarks.

---

## 1. Two structural findings that change the design

Both were derived from `data/variables.csv` (offline NatInst index + Task 1
counts) **before** any model was run.

### 1.1 §5.2 is not computable for CrowS-Pairs

Only **2 of 3,016** CrowS sentences are absent from the NatInst tasks, and
**0 of 1,508 pairs** have both members at zero dose. There is no uncontaminated
CrowS subset. The §5.2 headline sentence ("the reported bias score is X; on the
uncontaminated subset it is Y") **cannot be computed for CrowS-Pairs at all.**

It remains computable for StereoSet: 2,597 of 6,318 sentences are zero-dose
(stereo 920, anti-stereo 1,020, unrelated 657). §5.2 therefore becomes a
StereoSet-only result, and §9's "small zero-dose subset" risk is hereby declared
to have materialised in its severe form for CrowS.

### 1.2 §5.1 as specified does not identify the effect it claims to

The brief argues pair-level differencing "eliminates every sentence-level
confound." It does not eliminate the one that matters here. Δdose_natinst is
**+2 for 1,404 of 1,498 pairs** — sent_more carries two copies (task316 +
task317), sent_less carries one. The treatment is near-constant, and it is
applied in the *same direction as the construct*: the higher-dose member is
systematically the stereotypical one, which is exactly the member a biased model
is expected to prefer. **Dose and the bias signal are collinear within CrowS.**
A positive β is therefore consistent with both "exposure raises likelihood" and
"the model is biased," and §5.1 alone cannot separate them.

**Correction, pre-registered:** the causal weight moves to a
**difference-in-differences** across the §4.2 natural experiment, which is
available (confirmed: Dolma v1.5 contains neither the Flan Collection nor
StarCoder; v1.7 adds both; OLMo-7B trained on v1.5-sample, OLMo-7B-v1.7 on v1.7).
The estimand is:

```
beta_DiD = beta(Delta_dose, contaminated model) - beta(Delta_dose, clean model)
```

Identification rests on `dose_dolma` (which varies over 0–1728 and is not a
near-constant) rather than `dose_natinst`, and on the clean models acting as
placebos: a model never exposed to the injected copies must show no dose effect,
while the bias signal — being a property of the benchmark, not the corpus — is
present in both arms and differences out.

---

## 2. Pre-specified hypotheses, with committed sign and magnitude

Outcome = sentence log-likelihood. **Scorer A** = sum log-prob over scored tokens;
**Scorer B** = length-normalised mean log-prob. Every hypothesis is tested under
both; a result that holds under one only is reported as such.

| # | Hypothesis | Predicted sign | Predicted magnitude | My confidence |
|---|---|---|---|---|
| **H1** | β(Δlog(1+dose_dolma)) in contaminated OLMo-v1.7, within-pair | **positive** | small: +0.05 to +0.5 nats per unit Δlog-dose under Scorer A; **point estimate +0.2**. Under Scorer B, ≈ +0.02. | 60% |
| **H2** | β in clean arms (OLMo-7B v1.5, Pythia-6.9B) | **≈ zero** | \|β\| < 0.10 nats, CI covering 0 | 70% |
| **H3** | β_DiD = β(contaminated) − β(clean) | **positive** | +0.05 to +0.5, point estimate **+0.2** | 55% |
| **H4** | Sign agreement: P(sign Δdose_dolma = sign Δlogp) | **> 50%** | 52–62%, point estimate **55%** | 60% |
| **H5** | Effect is tail-driven, not uniform | high-dose decile shows larger effect | β among top-decile \|Δdose\| ≥ 2× β overall | 65% |
| **H6** | §5.2 StereoSet: full-benchmark stereotype score vs zero-dose subset | full **more** biased | gap **0–2 points**; point estimate **+0.8** on the 0–100 scale | 50% |
| **H7** | §5.3 ICAT: lm-score lower for high-dose `unrelated` items | **negative** relation | small; ICAT shift < 2 points | 50% |
| **H8** | 22 NatInst exemplar sentences show elevated logp vs pair partners | **positive** | descriptive only; n=22 cannot carry a regression | 70% |

**Magnitude reasoning, committed:** most sentences sit at 1–3 Dolma copies in a
2.6T-token corpus. Established memorisation curves imply almost nothing detectable
at that duplication level, so I expect the aggregate effect to be **small and
tail-driven** — carried by the handful of exemplar sentences at 800–1700 copies
(H5). I am explicitly predicting a *statistically real but practically modest*
effect, and per §6 a §5.2 gap under 1 point means this is a methods note rather
than a headline. H6's point estimate of +0.8 is below that line. I am recording
that I expect to land near the "real but trivial" boundary.

**Direction note.** Task 1 withdrew the original deflation hypothesis. I predict
contamination **inflates** apparent bias on CrowS (higher-dose member is the
stereotypical one). For StereoSet the picture is mixed, since `unrelated` is the
most-leaked role, which pushes on ICAT's LM term rather than the stereotype term.

---

## 3. Analysis plan (fixed in advance)

1. **Primary.** `Δlogp ~ Δlog(1+dose_dolma) + Δlog(1+dose_pile) + Δlength`,
   errors clustered by `bias_type`, fit separately per model × scorer. Then β_DiD
   across arms.
2. **Placebo.** Same regression on clean-arm models. A non-null here invalidates
   the design and will be reported as such.
3. **Secondary.** Sign-agreement rate; tail/decile analysis (H5); StereoSet
   zero-dose recomputation (H6) with bootstrap CI over 10,000 resamples;
   exemplar case study (H8).
4. **Covariates** as §3.3. `is_natinst_exemplar` is **descriptive only** — n=22
   total (10 with dose so far), too few to enter a regression.

## 4. Stopping rules

- H1 positive and significant under **both** scorers, **and** H2 null → proceed to Phase B.
- H1 null or negative under either scorer → **stop, report the null.**
- H2 non-null (clean arms show a dose effect) → design is broken; debug before any claim.
- H6 gap < 1 point → reframe as methods note per §6, do not headline.

## 5. Deviations register

Any departure from this file will be listed here in `summary.md` with a reason.
