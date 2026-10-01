"""
THE CORE PIVOT: this is no longer "rank the diagnosis for one visit."

The real-world failure mode this targets is different and, I'd argue,
more important: a patient sees a provider, reports some symptoms, gets
a plausible working diagnosis, and goes home. Weeks later they're back
- maybe with the same provider, maybe not - with a few more symptoms
that have emerged. Each individual visit can look completely
reasonable in isolation. The miss happens because nobody is looking
at the FULL picture across visits - which is exactly the fragmentation
problem a platform that "aggregates data across the continuum of care"
(Lumeris's own description of what it does) is positioned to solve.

This module simulates that: synthetic patients whose symptoms emerge
gradually across multiple visits (not all on day one - realistic,
since patients rarely report every symptom at the first visit), and
compares two diagnostic strategies at each visit:

  BASELINE  ("fragmented care"): diagnose using only the symptoms
            reported AT that single visit - as if each visit is seen
            in isolation, which is what happens by default when
            encounter data isn't aggregated across providers/visits.

  LONGITUDINAL ("aggregated care"): diagnose using the FULL symptom
            history accumulated across all visits so far.

Both use the exact same trained classifier (diagnosis_classifier.py) -
the only difference is whether it sees one visit's symptoms or the
patient's whole history. That isolates the thing actually being
tested: does aggregating encounters help, independent of model
quality.

Metric: how many visits (and, as a rough illustrative estimate, how
many days) earlier does the longitudinal approach converge on the
correct diagnosis, compared to the fragmented baseline - and how often
does the fragmented baseline fail to converge at all within a
realistic visit window while the longitudinal approach succeeds.
"""
import random
import pickle
import numpy as np
from vignette_generator import load_knowledge_base, ALL_SYMPTOMS
from diagnosis_classifier import DiagnosisClassifier, vectorize

MAX_VISITS = 5
DAYS_BETWEEN_VISITS = 18  # illustrative estimate only - see README caveat


def simulate_patient_symptom_reveal(disease: dict, rng):
    """The patient's eventual full symptom set (same noisy draw as
    vignette_generator), then shuffled into the order symptoms get
    REPORTED across visits - modeling that patients don't report
    everything at visit 1."""
    full_symptoms = []
    for s, weight in disease["symptoms"].items():
        if rng.random() < weight:
            full_symptoms.append(s)
    for s in ALL_SYMPTOMS:
        if s not in disease["symptoms"] and rng.random() < 0.08:
            full_symptoms.append(s)
    rng.shuffle(full_symptoms)

    # distribute across MAX_VISITS visits - at least 1 symptom at visit 1,
    # remaining symptoms trickle in over subsequent visits
    if not full_symptoms:
        return [[] for _ in range(MAX_VISITS)]
    per_visit = [[] for _ in range(MAX_VISITS)]
    for i, symptom in enumerate(full_symptoms):
        visit_idx = min(i // max(1, len(full_symptoms) // MAX_VISITS + 1), MAX_VISITS - 1)
        per_visit[visit_idx].append(symptom)
    return per_visit


def run_simulation(clf: DiagnosisClassifier, kb, n_patients_per_pair=150, seed=7):
    """
    BASELINE now models actual anchoring bias, not a strawman: the
    working diagnosis is set at visit 0 from whatever symptoms have
    appeared so far, and then STAYS FIXED on subsequent visits unless a
    new symptom appears that has near-zero weight in the currently-
    anchored diagnosis's own profile (i.e., a symptom that clearly
    doesn't fit the current story) - which forces a fresh look at the
    full cumulative picture. This is the actual clinical phenomenon:
    new, ambiguous symptoms get folded into the existing diagnosis
    rather than triggering a full re-differential each time; only a
    clearly-inconsistent finding forces reconsideration.

    LONGITUDINAL has no anchor: every visit, it reclassifies from the
    full cumulative symptom history with no stickiness to the prior
    visit's answer.

    Both strategies see the exact same symptom-reveal schedule for a
    given simulated patient - the only difference being tested is
    anchoring vs. continuous full-history reconsideration.

    The anchoring mechanism modeled here isn't invented - it's a
    simplified version of documented automation-bias/trust dynamics:
    Goddard K, Roudsari A, Wyatt JC. Automation bias: a systematic
    review of frequency, effect mediators, and mitigators. JAMIA.
    2012;19(1):121-127. And Lyell D, Coiera E. Automation bias and
    verification complexity: a systematic review. JAMIA.
    2017;24(2):423-431 (covers both over-trust and the under-trust
    failure mode this simulation's "poor-fit symptom" reconsideration
    trigger is a stand-in for).
    """
    rng = random.Random(seed)
    name_to_disease = {d["name"]: d for d in kb}
    pairs = []
    seen = set()
    for d in kb:
        for other in d.get("commonly_confused_with", []):
            if other not in name_to_disease:
                continue  # e.g. "Normal Aging" - not a disease profile in the KB
            key = tuple(sorted([d["name"], other]))
            if key not in seen:
                seen.add(key)
                pairs.append(key)

    results = []
    for true_name, other_name in pairs:
        disease = name_to_disease[true_name]
        for _ in range(n_patients_per_pair):
            per_visit_new_symptoms = simulate_patient_symptom_reveal(disease, rng)

            cumulative = []
            baseline_anchor = None  # current working diagnosis name, once set
            baseline_catch = None
            longitudinal_catch = None

            for visit_idx in range(MAX_VISITS):
                new_this_visit = per_visit_new_symptoms[visit_idx]
                cumulative = cumulative + new_this_visit

                # --- longitudinal: fresh full-history read every visit ---
                if cumulative and longitudinal_catch is None:
                    pred = clf.predict_proba_ranked(cumulative, top_n=1)
                    if pred and pred[0][0] == true_name:
                        longitudinal_catch = visit_idx

                # --- baseline: anchored, only reconsiders on a poor-fit symptom ---
                if baseline_anchor is None and cumulative:
                    pred = clf.predict_proba_ranked(cumulative, top_n=1)
                    baseline_anchor = pred[0][0] if pred else None
                elif new_this_visit and baseline_anchor is not None:
                    anchor_profile = name_to_disease.get(baseline_anchor, {}).get("symptoms", {})
                    poor_fit = any(anchor_profile.get(s, 0) < 0.15 for s in new_this_visit)
                    if poor_fit:
                        pred = clf.predict_proba_ranked(cumulative, top_n=1)
                        baseline_anchor = pred[0][0] if pred else baseline_anchor

                if baseline_catch is None and baseline_anchor == true_name:
                    baseline_catch = visit_idx

            results.append({
                "true_disease": true_name,
                "confused_with": other_name,
                "baseline_catch_visit": baseline_catch,
                "longitudinal_catch_visit": longitudinal_catch,
            })
    return results


def summarize(results):
    caught_both = [r for r in results
                   if r["baseline_catch_visit"] is not None and r["longitudinal_catch_visit"] is not None]
    visits_earlier = [r["baseline_catch_visit"] - r["longitudinal_catch_visit"] for r in caught_both]

    baseline_never_caught = sum(1 for r in results if r["baseline_catch_visit"] is None)
    longitudinal_never_caught = sum(1 for r in results if r["longitudinal_catch_visit"] is None)
    longitudinal_rescued = sum(1 for r in results
                               if r["baseline_catch_visit"] is None and r["longitudinal_catch_visit"] is not None)

    return {
        "n_patients": len(results),
        "avg_visits_earlier": float(np.mean(visits_earlier)) if visits_earlier else 0.0,
        "avg_days_earlier": float(np.mean(visits_earlier)) * DAYS_BETWEEN_VISITS if visits_earlier else 0.0,
        "pct_baseline_never_caught": baseline_never_caught / len(results) * 100,
        "pct_longitudinal_never_caught": longitudinal_never_caught / len(results) * 100,
        "n_rescued_by_longitudinal": longitudinal_rescued,
        "pct_rescued": longitudinal_rescued / len(results) * 100,
    }


def summarize_by_pair(results):
    pairs = sorted(set((r["true_disease"], r["confused_with"]) for r in results))
    rows = []
    for true_name, other_name in pairs:
        subset = [r for r in results if r["true_disease"] == true_name and r["confused_with"] == other_name]
        s = summarize(subset)
        rows.append((true_name, other_name, s))
    return rows


if __name__ == "__main__":
    kb = load_knowledge_base()
    with open("trained_classifier.pkl", "rb") as f:
        clf = pickle.load(f)

    results = run_simulation(clf, kb, n_patients_per_pair=150)
    overall = summarize(results)

    print("=" * 70)
    print("LONGITUDINAL (AGGREGATED-ENCOUNTER) vs FRAGMENTED (PER-VISIT) CARE")
    print("=" * 70)
    print(f"Simulated patients: {overall['n_patients']} across all commonly-confused pairs")
    print(f"\nAverage visits earlier the correct diagnosis is reached: "
          f"{overall['avg_visits_earlier']:.2f}")
    print(f"Illustrative days earlier (assuming ~{DAYS_BETWEEN_VISITS}-day visit interval): "
          f"{overall['avg_days_earlier']:.0f} days")
    print(f"\n% of cases where per-visit-only care NEVER reaches the correct "
          f"diagnosis within {MAX_VISITS} visits: {overall['pct_baseline_never_caught']:.1f}%")
    print(f"% of cases where aggregated-history care never reaches it either: "
          f"{overall['pct_longitudinal_never_caught']:.1f}%")
    print(f"-> Cases where aggregating history RESCUES a diagnosis fragmented "
          f"care would have missed entirely: {overall['n_rescued_by_longitudinal']} "
          f"({overall['pct_rescued']:.1f}%)")

    print("\n--- Breakdown by commonly-confused pair ---")
    for true_name, other_name, s in summarize_by_pair(results):
        print(f"{true_name} (vs {other_name}): "
              f"{s['avg_visits_earlier']:+.2f} visits earlier, "
              f"{s['pct_rescued']:.1f}% rescued from never-caught")
