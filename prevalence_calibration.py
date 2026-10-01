"""
THE GAP THIS FIXES (found by applying HINF 5500 Week 1, Knowledge
Check 2, Question 1 to this project's own classifier):

diagnosis_classifier.py was trained on PERFECTLY BALANCED synthetic
data - 200 vignettes per disease, 13 diseases. A classifier trained
on balanced classes implicitly assumes every disease is equally
likely before symptoms are considered (~7.7% prior each, 1/13). That
assumption is silently baked into every probability the model
outputs - it is not a neutral default, it is a specific, wrong claim
about the world.

Sensitivity and specificity are properties of the model. PPV - "given
this score, what's the actual probability this patient has the
disease" - is a property of the model PLUS the population's true
prevalence. A model can report 94% confidence in Pulmonary Embolism
and be making a defensible claim ONLY if PE really is as common as
Anxiety Disorder among patients with this symptom cluster. It is not.

This module re-weights the model's raw output using Bayes' rule and
a realistic prior, instead of the implicit uniform one the balanced
training data assumed. This is exactly the PPV-collapse arithmetic
from the course (sepsis alert, lung nodule screening, AKI model) -
same mechanism, applied to this project's own classifier before
anyone else had to point it out.
"""
import pickle
from vignette_generator import load_knowledge_base
from diagnosis_classifier import DiagnosisClassifier  # noqa: F401 (needed for unpickling)

UNIFORM_PRIOR = 1 / 13  # what the balanced training data implicitly assumes per disease


def recalibrate(raw_ranked: list, realistic_priors: dict, uniform_prior=UNIFORM_PRIOR):
    """
    raw_ranked: [(disease_name, raw_confidence), ...] from the trained
                classifier - confidences that implicitly assume a
                uniform 1/13 prior per disease.
    realistic_priors: {disease_name: true_prevalence_in_this_population}
                       for diseases you have real epidemiological data
                       for. Diseases not in this dict keep their raw
                       (uniform-prior) value, renormalized at the end.

    Returns the same ranked list, re-sorted, with confidences adjusted
    toward what the real population's prevalence would actually
    produce - via Bayes' rule: posterior_new proportional to
    (raw / uniform_prior) * realistic_prior.
    """
    adjusted = []
    for name, raw_conf in raw_ranked:
        if name in realistic_priors:
            likelihood_ratio = raw_conf / uniform_prior if uniform_prior > 0 else 0
            new_conf = likelihood_ratio * realistic_priors[name]
        else:
            new_conf = raw_conf
        adjusted.append((name, new_conf))

    total = sum(c for _, c in adjusted) or 1.0
    adjusted = [(name, c / total) for name, c in adjusted]
    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted


if __name__ == "__main__":
    with open("trained_classifier.pkl", "rb") as f:
        clf = pickle.load(f)

    symptoms = ["shortness_of_breath", "chest_pain", "rapid_heartrate"]
    raw = clf.predict_proba_ranked(symptoms, top_n=5)

    print("=" * 70)
    print("RAW MODEL OUTPUT (implicit uniform 7.7% prior, from balanced training)")
    print("=" * 70)
    for name, conf in raw:
        print(f"  {name}: {conf:.1%}")

    # Illustrative realistic priors for patients presenting with THIS
    # symptom cluster (dyspnea + chest pain + tachycardia) in a general
    # ED population - NOT this project's invention. PE prevalence
    # among patients worked up for suspected PE is well documented at
    # roughly 7-13% in North American ED cohorts (e.g. Hammer et al.,
    # ~8.9% at one urban teaching hospital; commonly cited 7-13% range
    # across US/Canadian studies). Anxiety/panic presentations with
    # this exact symptom triad are substantially more common in
    # undifferentiated ED chest-pain populations. These relative
    # weights are illustrative of the DIRECTION and MAGNITUDE of the
    # correction, not a validated prevalence table for any specific
    # institution's population - that number has to come from local
    # epidemiological data, per the course's own governance point.
    realistic_priors = {
        "Pulmonary Embolism": 0.10,
        "Anxiety Disorder": 0.50,
        "Asthma": 0.25,
        "COPD": 0.10,
        "Pneumonia": 0.05,
    }

    calibrated = recalibrate(raw, realistic_priors)
    print("\n" + "=" * 70)
    print("CALIBRATED OUTPUT (illustrative realistic priors - see source note above)")
    print("=" * 70)
    for name, conf in calibrated:
        print(f"  {name}: {conf:.1%}")

    print("\nThe raw model's 94% confidence in Pulmonary Embolism is not a "
          "deployable number - it is an artifact of balanced training data. "
          "The calibrated figure is still illustrative, not validated, but it "
          "is honest about WHY it differs from the raw output, which the raw "
          "output alone is not.")
