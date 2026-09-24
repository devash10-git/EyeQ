# Multi-participant validation study — protocol

Draft for the study that converts the paper's single-participant result into a
population-scale one. Written to be run as-is: the capture tooling
(`eval/capture.html`, `eval/capture.py`) already produces the session format the
analysis scripts consume.

**Status:** not yet run. Nothing in this document is a result.

---

## 1. What this study is for

Three specific weaknesses in the current paper, in order of how much they cost:

| Weakness | Fixed by |
|---|---|
| Canthus effect rests on 3 sessions, 1 participant | Block A, across participants |
| Table V roll budget rests on 3 fixations, 1 session | Block C |
| Eyelid mechanism is inferred, never demonstrated | Block D |

Block D is the one that changes the paper's standing. The canthus result is
currently an *association* — the feature set that avoids the eyelid performs
better. Block D makes it *causal* by perturbing the eyelid directly and
predicting, in advance, which feature set degrades.

## 2. Pre-specified hypotheses

Registered before collection; the analysis plan in §8 is fixed at the same time.

- **H1 (primary).** Canthus-relative normalisation reduces held-out gaze error
  relative to eyelid-relative normalisation, within participant.
  *Test:* paired two-tailed t-test on per-participant mean error, α = 0.05.
- **H2 (causal mechanism).** Under deliberate eyelid narrowing at a fixed gaze
  target, eyelid-relative features degrade significantly more than
  canthus-relative features.
  *Test:* interaction term, condition × feature set, repeated-measures ANOVA.
- **H3 (roll).** The within-fixation share of total error increases with
  |roll|. *Test:* per-participant Spearman correlation between binned |roll|
  and noise share, then a one-sample t-test on the Fisher-z values.
- **H4 (calibration size).** The ridge-versus-kernel-ridge ranking inverts
  between 9 and 20 calibration targets. *Test:* paired comparison at each size,
  as in §VII-D of the paper.

H2 is the one with a directional prediction that could fail cleanly, which is
what makes it worth running.

## 3. Sample size

Grounded in the observed effect rather than a convention. The three recorded
sessions give paired differences of 1.02°, 3.46° and 0.69° (mean 1.72°,
sd 1.51°), so Cohen's *dz* ≈ 1.14.

| n | power at *dz* = 1.14 | power if the true effect is half |
|---|---|---|
| 8 | 0.79 | 0.28 |
| 10 | 0.89 | 0.36 |
| **12** | **0.95** | 0.44 |
| 20 | >0.99 | 0.68 |
| 24 | >0.99 | 0.76 |

Three sessions from one person almost certainly overstate the effect, so the
right target is set by the conservative column.

**Recruit 20, analyse a minimum of 12.** Twelve gives 95% power if the effect
holds at the observed size; twenty keeps the study informative if it is half
that. Budget for ~15% exclusion (§7), so recruit 24 to land 20.

## 4. Participants

- Adults 18+, normal or corrected-to-normal vision.
- **Stratify deliberately on what the method claims robustness to**, or the
  study cannot test that claim: recruit for a spread of eye shape and
  inter-canthal width, include spectacle wearers (target ≥ 6), and include at
  least 4 participants with epicanthic folds, since the canthus landmark is
  exactly what varies there.
- Exclude anyone with nystagmus, strabismus or an eye condition affecting
  fixation stability — not because the system should fail, but because these
  need their own study rather than being 2 of 20 here.
- No compensation tied to performance.

## 5. Apparatus

One machine, one display, one camera for every participant. Record the display
diagonal, resolution and the seated eye-to-screen distance per session; the
degrees conversion is measured, never assumed.

Fixed 60 cm viewing distance with a chin rest **for Blocks A, B and D**. Block C
requires free head movement by design. Ambient illumination 300–500 lx at the
face, measured and logged, with the webcam's auto-exposure disabled.

## 6. Session structure

Roughly 35 minutes per participant. Two sessions each, ≥ 24 h apart, so
between-session variance is estimable — that is what the current paper cannot
do with one participant.

**Block A — standard calibration and validation (12 min).**
21 targets, 5 fixations each, 300 ms retained per fixation after discarding the
first 150 ms. 13 targets for fitting, 8 held out. This is the existing protocol
unchanged, and it supplies H1 and H4.

**Block B — calibration density (6 min).**
A second pass collecting 20 targets so the calibration-size sweep can be run
per participant rather than by subsampling one session. Supplies H4.

**Block C — head roll (8 min).**
Targets held at instructed roll angles: 0°, ±5°, ±10°, ±15°, with 8 fixations
per angle per participant. A printed angle guide at eye level; the achieved
angle is *measured* from the landmarks and used for binning, never the
instructed one. This is what Table V needs and currently lacks — the present
data has 3 fixations above 10°, all from one session.

**Block D — eyelid perturbation (9 min).**
The decisive test, and the block to protect if time runs short. Gaze target is
held **fixed at screen centre** throughout, so any feature change is attributable
to the eyelid and not to gaze. Four conditions, 10 fixations each, order
counterbalanced across participants:

1. neutral,
2. deliberate squint (instructed narrowing, held),
3. wide-open (instructed raising of the upper lid),
4. natural blink recovery (measured in the 500 ms after a blink).

Eye aspect ratio is logged per frame as the manipulation check: if EAR does not
separate the conditions, the manipulation failed and the block is void for that
participant.

**Prediction, stated now:** eyelid-relative vertical features shift
significantly across conditions 1–3 while canthus-relative features do not.
Failure to observe that would falsify the mechanism the paper argues for, and we
would report it.

## 7. Exclusions — fixed in advance

A session is excluded if it loses > 20% of frames to tracking dropout, fails to
cover the full target grid, or (Block D) shows no EAR separation between
conditions. Exclusions are reported with reasons and counts. One session in the
current data lost 67% of frames to poor illumination and was excluded; that is
the precedent.

## 8. Analysis plan — fixed before collection

- **Unit of analysis is the participant**, not the frame. Per-participant means
  enter the group test. Frames within a participant are not independent and
  treating them as such inflates n by three orders of magnitude.
- All splits grouped by target, as in §VI-B of the paper.
- H1: paired t-test on per-participant mean error, plus the per-participant
  effect sizes so heterogeneity is visible rather than averaged away.
- H2: repeated-measures ANOVA, feature set × condition, with the interaction as
  the test of interest.
- H3: per-participant Spearman, Fisher-z, one-sample t-test.
- H4: paired comparison at 9 and 20 targets, Wilcoxon signed-rank.
- Report effect sizes and confidence intervals, not just p-values. No
  subgroup analysis that was not pre-specified here.

## 9. Ethics and data handling

Institutional ethics approval before any collection. Written informed consent
covering webcam recording, what is stored, and withdrawal rights.

**Store landmarks, not video.** The pipeline needs only the 478 landmark
coordinates per frame, which are not biometrically identifying in the way a face
image is, and this removes the largest privacy risk at no analytical cost. If
video must be retained for re-analysis, store it separately under explicit
consent with a stated deletion date.

Participants identified by code only. Consent forms stored apart from data.

## 10. Reporting

Report all 20 participants, including those where the effect is absent or
reversed. Per-participant results in an appendix table. Pre-registration of
§2 and §8 before collection, referenced in the paper.

---

## What this does and does not do for the paper

**Does:** replaces the single-participant caveat with a population estimate;
turns the eyelid mechanism from inference into evidence; gives Table V real
statistical power; lets the calibration-size finding be stated across people.

**Does not:** add novelty on its own. A protocol that has not been run is not a
result, and describing it in the paper strengthens credibility, not
contribution. The paper's novel claims are the canthus normalisation, the
diagnosed negative result, and the calibration-size ranking inversion — the
study raises confidence in all three rather than adding a fourth.
