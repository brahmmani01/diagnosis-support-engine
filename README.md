# Diagnostic Continuity Engine (v3) — formerly a differential-diagnosis ranker

**The pivot that matters:** v1 and v2 of this project ranked diagnoses
for a single visit. That's a crowded category — nearly every health-AI
demo day has one. v3 asks a different, less-pitched question: **what
if the diagnosis is actually correct in principle, but gets missed
because nobody is looking at the patient's full history across
visits?** That's a documented, named failure mode in patient-safety
research (Hardeep Singh's SPADE framework — Symptom-Disease Pair
Analysis of Diagnostic Error), and it's also exactly the gap a
platform built to **"aggregate data across the continuum of care"**
(Lumeris's own description of its model) is positioned to close.

## The headline result

Simulating patients whose symptoms emerge gradually across up to 5
visits (not all at once — realistic, since patients rarely report
everything at visit one), comparing two diagnostic strategies using
the *same* trained classifier:

- **Fragmented, anchored care**: a working diagnosis is set early and
  stays fixed across visits unless a clearly inconsistent symptom
  forces reconsideration — modeling real anchoring bias, not a
  strawman of "no memory at all."
- **Aggregated care**: the full cumulative symptom history is
  reconsidered fresh at every visit.

Across 1,800 simulated patients spanning every commonly-confused
disease pair in the knowledge base:

| | Fragmented (anchored) | Aggregated (longitudinal) |
|---|---|---|
| Never reaches correct diagnosis within 5 visits | **32.7%** | **25.0%** |
| → Cases *rescued* by aggregating history | — | **139 patients (7.7%)** |

That rescue rate held in **every single confused-disease pair tested**
(see `figures/longitudinal_rescue_by_pair.png`), not just on average —
the strongest kind of result to hold up to pointed follow-up
questions, because it isn't hiding a reversal in some subgroup.

## Why this, specifically, for Lumeris

Lumeris's own public materials describe exactly this gap: AI that
"may suggest information... that might inform a diagnosis, but
ultimately, a clinician is doing that" — augmentation, not automation.
A tool that flags *"this patient's pattern across their last 4 visits
is more consistent with X than the working diagnosis of Y — worth a
second look"* is squarely inside that philosophy, and it's also a
tool whose value is structurally tied to **value-based care economics**:
catching a missed diagnosis earlier reduces the downstream cost of a
late-stage complication, which is the exact mechanism by which
Lumeris profits when a health system succeeds.

## Architecture (builds on v2)

```
per-visit clinical notes (free text, across multiple encounters)
    |
    v
nlp_extractor.py  -- symptom extraction per visit (negation-aware)
    |
    v
longitudinal_simulation.py  -- THE NEW PIECE:
    |    - accumulates symptoms across visits
    |    - re-evaluates the full history at each encounter
    |    - flags when the aggregated picture diverges from the
    |      currently-anchored working diagnosis
    v
diagnosis_classifier.py  -- same trained model as v2, just fed
    |                        accumulated history instead of one visit
    v
fhir_adapter.py  -- FHIR DiagnosticReport output
```

v2's single-visit ranker (`diagnosis_classifier.py`, `pipeline.py`)
is not replaced — it's the building block this simulation calls at
each visit. The pivot is in *what gets fed to it* (one visit vs. full
history) and *what gets measured* (visits-to-correct-diagnosis, not
single-shot accuracy).

## Running it

```bash
python3 vignette_generator.py          # synthetic vignettes
python3 diagnosis_classifier.py        # trains + saves the classifier
python3 longitudinal_simulation.py     # THE headline result - run this one
python3 tests/test_longitudinal_simulation.py   # 3 tests, including a
                                                 # strict per-subgroup check
```

## What this is honestly not (yet)

- **A simulation, not real EHR evidence.** Every "patient" here is
  synthetic, built from the knowledge base's own symptom-weight
  profiles. The result shows the *mechanism* (aggregating evidence
  across encounters helps, anchoring bias has a measurable cost) works
  as designed — it is not a claim about real-world diagnostic delay
  reduction until validated against real, de-identified longitudinal
  EHR data.
- **The anchoring model is simplified.** Real anchoring bias involves
  more than "ignore new symptoms unless they clearly don't fit" — social
  and system factors (visit time pressure, handoffs between different
  providers, documentation burden) all contribute and aren't modeled
  here.
- **Visit-interval-to-days conversion is illustrative only** (assumes
  ~18 days between visits for an unresolved complaint) — a real
  deployment would use actual visit-timestamp data, not an assumption.

## What would make this real

- Replace simulated visit sequences with real (de-identified,
  IRB-approved) longitudinal encounter data
- Validate the "poor-fit symptom forces reconsideration" anchoring
  model against how clinicians actually update differentials in
  practice (chart review study, not assumption)
- Extend from the current 13-disease knowledge base to a
  production-scale set of confusable diagnosis pairs, ideally learned
  from retrospective diagnostic-error data (e.g. malpractice closed
  claims analysis, the data source Singh's own SPADE work draws on)
- Tie the "rescue" cases to an actual cost/outcome estimate (e.g.
  avoided ED visit, avoided late-stage diagnosis) rather than a
  visit-count proxy

## Everything from v2 (still included, still true)

- Free-text symptom extraction with negation handling
  (`nlp_extractor.py`)
- A real `ClinicalBERTSymptomExtractor` class (written, documented,
  not executed here — no internet/GPU in this sandbox; see
  instructions further down for running it in Colab)
- FHIR-shaped input/output with ICD-10 codes (`fhir_adapter.py`)
- Trained, evaluated single-visit classifier: 65.4% top-1 / 92.6%
  top-3 accuracy on held-out synthetic data (`diagnosis_classifier.py`)

### Running the ClinicalBERT extractor

`ClinicalBERTSymptomExtractor` in `nlp_extractor.py` needs
`transformers` + `torch` and internet access to download model
weights (~400MB) — not available in this sandbox, so it's written but
unbenchmarked here:

1. Open **Google Colab** (free GPU): https://colab.research.google.com
2. `!pip install transformers torch`
3. Upload `nlp_extractor.py` and `vignette_generator.py`
4. ```python
   from nlp_extractor import ClinicalBERTSymptomExtractor
   extractor = ClinicalBERTSymptomExtractor()
   extractor.extract("Pt c/o fatigue, joint pain, low-grade fevers x3 weeks")
   ```

## Team

Original concept: Brahmmani Thota, Karthik Raj Sunder Raj, Kenny
Serubiri, Vishruti Savaj (course pitch, "Accuview"). v2/v3 engineering
rebuild and the continuity-of-care pivot: Brahmmani Thota.

## Governance readiness

Applying the evaluation framework from Northeastern's HINF 5500 (AI in
Health Informatics) to this project's own code, honestly, rather than
only to vendor examples in the course material — grounded in the
primary literature below, not just lecture paraphrase.

### PPV collapses under realistic prevalence — the gap this surfaced
`diagnosis_classifier.py` trains on **perfectly balanced** synthetic
data (200 vignettes/disease), which implicitly assumes every disease
is equally likely (~7.7% prior). That assumption is silently baked
into every confidence score the model outputs — the same mechanism
by which **Wong et al. (2021)** found a widely-deployed proprietary
sepsis model's real-world performance diverged sharply from its
internal validation once measured at external sites: a model's
reported numbers reflect the population it was built on, not the one
it's pointed at. `prevalence_calibration.py` fixes this: re-weighting
the PE-vs-anxiety case against illustrative real-world priors drops
the model's raw **94% confidence in Pulmonary Embolism to 78.8%**,
with Anxiety Disorder's share rising from 4.7% to 19.6%. The raw
number was never deployable; it was an artifact of balanced training
data, not a claim about any real population. (**Sjoding et al. 2020**
is the sharpest analog for how an unexamined assumption embedded
upstream — there, pulse oximetry calibration; here, class balance —
propagates invisibly into everything built on top of it.)

### Where this stands on the four accountability senses
Framework: **Habli, Lawton & Porter (2020)**, "Artificial intelligence
in health care: accountability and safety," *Bull World Health Organ*.
- **Causal accountability**: partial. `fhir_adapter.py` outputs a
  `DiagnosticReport`, but does not yet log model version, exact input
  vector, and timestamp together — the audit trail a real incident
  review would need. Not yet built.
- **Role accountability**: unassigned. No named owner for monitoring
  subgroup performance, setting the confidence threshold, or
  authorizing deployment — this is an institutional decision, not a
  model property, and this project doesn't get to make it.
- **Remedial accountability**: unaddressed. What a patient is owed if
  this system contributes to a missed or delayed diagnosis is not
  specified anywhere in this repo.
- **Legal accountability**: out of scope for a prototype; would be a
  contract term, not a code artifact.

### Contestability
The patient knowing AI was involved, accessing what it produced and on
what basis, and a human-review mechanism (**Mittelstadt et al. 2016**;
**Floridi et al. 2018**, the AI4People framework) — **this project
satisfies the second condition only** (the explanation output shows
matched/missing symptoms and feature attribution). The first and third
are deployment decisions, not something a differential-diagnosis
script can guarantee on its own.

### Surrogate vs. clinical endpoint
"Visits earlier the correct diagnosis is reached" is a **surrogate
endpoint** for what actually matters — patient outcomes. A faster
correct diagnosis is not automatically a better outcome unless it
changes treatment in time to matter — precisely the gap **Kelly et
al. (2019)** and **Nagendran et al. (2020)** identify across the
clinical-AI evidence base generally: reported performance gains
routinely fail to translate into demonstrated clinical impact. This
project has not measured that, and says so rather than implying the
rescue-rate number is an outcomes claim.

### Beneficence — which level this project has shown
Framework: **Beauchamp & Childress**, *Principles of Biomedical
Ethics* (the canonical four-principles source). **Level 1 (technical)
only.** The classifier's accuracy and the simulation's rescue rate are
technical/mechanistic results. Level 2 (clinical pathway — does a
flagged inconsistency actually lead to a changed clinical action) has
one documented positive design pattern to emulate: **Sendak et al.
(2020)**'s account of Duke's Sepsis Watch routes alerts to a staffed
nurse triage hub with real authority to act, rather than a dashboard
no one owns — exactly the gap between detection and action this
project's own simulation targets. Level 3 (distributional — does the
benefit reach every subgroup) is unmeasured here, and **Obermeyer et
al. (2019)** is the canonical warning for why that can't be assumed:
a proxy variable chosen for technical convenience encoded racial bias
invisibly, at scale, with aggregate accuracy metrics that never
surfaced it.

### The anchoring model itself has an evidence base
`longitudinal_simulation.py`'s "fragmented care" baseline — a working
diagnosis set early and defended unless contradicted — isn't an
invented mechanism. It's a simplified version of documented automation
bias and threshold-dependent trust dynamics: **Goddard, Roudsari &
Wyatt (2012)** and **Lyell & Coiera (2017)** both document that
clinician trust in a system (over-trust AND under-trust) is shaped by
the information environment the system creates, not by individual
diligence — the same point the course material makes about alert
fatigue being arithmetic rather than a behavior problem.

### What this means in one sentence
Every honest gap above is a reason this is a **prototype demonstrating
a mechanism**, not a deployable governance-ready system — and saying
that plainly, grounded in the literature rather than hand-waved, is a
stronger claim than pretending otherwise.

### References
- Wong A, et al. External Validation of a Widely Implemented
  Proprietary Sepsis Prediction Model. *JAMA Intern Med*.
  2021;181(8):1065-1070.
- Sjoding MW, et al. Racial Bias in Pulse Oximetry Measurement. *NEJM*.
  2020;383:2477-2478.
- Habli I, Lawton T, Porter Z. Artificial intelligence in health care:
  accountability and safety. *Bull World Health Organ*.
  2020;98(4):251-256.
- Mittelstadt BD, et al. The ethics of algorithms: mapping the debate.
  *Big Data Soc*. 2016;3(2).
- Floridi L, et al. AI4People — an ethical framework for a good AI
  society. *Minds Mach*. 2018;28:689-707.
- Kelly CJ, et al. Key challenges for delivering clinical impact with
  artificial intelligence. *BMC Med*. 2019;17:195.
- Nagendran M, et al. Artificial intelligence versus clinicians.
  *BMJ*. 2020;368:m689.
- Beauchamp TL, Childress JF. *Principles of Biomedical Ethics* (8th
  ed.). Oxford University Press; 2019.
- Sendak MP, et al. A Path for Translation of Machine Learning
  Products into Healthcare Delivery. *NEJM Catalyst Innov Care
  Deliv*. 2020;1(3).
- Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial
  bias in an algorithm used to manage the health of populations.
  *Science*. 2019;366(6464):447-453.
- Goddard K, Roudsari A, Wyatt JC. Automation bias: a systematic
  review of frequency, effect mediators, and mitigators. *JAMIA*.
  2012;19(1):121-127.
- Lyell D, Coiera E. Automation bias and verification complexity: a
  systematic review. *JAMIA*. 2017;24(2):423-431.
- Chapman WW, et al. A Simple Algorithm for Identifying Negated
  Findings and Diseases in Discharge Summaries. *J Biomed Inform*.
  2001;34(5):301-310. (basis for `nlp_extractor.py`'s negation
  handling)
- Larson DB, et al. Regulatory Frameworks for Development and
  Evaluation of Artificial Intelligence-Based Diagnostic Imaging
  Algorithms. *J Am Coll Radiol*. 2021;18(3):413-424. (510(k) vs.
  clinical-validation gap, cited in `fhir_adapter.py`'s design intent)
