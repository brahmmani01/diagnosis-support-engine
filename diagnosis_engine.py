"""
A transparent, rule-based differential-diagnosis scoring engine —
a simplified, runnable stand-in for the "AI-Driven Differential
Diagnosis and Misdiagnosis Prevention" system pitched in the
accompanying course project slides (Accuview 17.4 concept).

The pitch described an NLP + supervised-learning system trained on
MIMIC-III / PubMed with explainable confidence scores. This module
implements the same *interface* (patient symptoms in -> ranked,
explained differential diagnoses out, with a bias-detection flag)
using a transparent weighted symptom-overlap score instead of a
trained model, so the reasoning is fully inspectable and the project
is runnable without a clinical dataset or GPU.

Real deployment would replace `score_disease()` with a trained
classifier (e.g. gradient boosting or a fine-tuned clinical language
model) while keeping the same ranked-output + explanation contract.
"""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiagnosisResult:
    disease: str
    confidence: float
    matched_symptoms: list
    missing_key_symptoms: list


def load_knowledge_base(path="knowledge_base.json"):
    with open(path) as f:
        return json.load(f)["diseases"]


def score_disease(patient_symptoms: set, disease: dict) -> DiagnosisResult:
    """Weighted overlap between reported symptoms and a disease's symptom
    profile, normalized so confidence is comparable across diseases with
    different numbers of hallmark symptoms."""
    profile = disease["symptoms"]
    matched = {s: w for s, w in profile.items() if s in patient_symptoms}
    max_possible = sum(profile.values())
    raw_score = sum(matched.values())
    confidence = raw_score / max_possible if max_possible > 0 else 0.0
    missing_key = [s for s, w in profile.items() if w >= 0.7 and s not in patient_symptoms]
    return DiagnosisResult(
        disease=disease["name"],
        confidence=round(confidence, 3),
        matched_symptoms=sorted(matched, key=matched.get, reverse=True),
        missing_key_symptoms=missing_key,
    )


def rank_differentials(patient_symptoms: list, knowledge_base=None, top_n=5):
    if knowledge_base is None:
        knowledge_base = load_knowledge_base(
            str(Path(__file__).parent / "knowledge_base.json"))
    patient_set = set(patient_symptoms)
    results = [score_disease(patient_set, d) for d in knowledge_base]
    results = [r for r in results if r.confidence > 0]
    results.sort(key=lambda r: r.confidence, reverse=True)
    return results[:top_n]


def detect_anchoring_bias(results: list, margin: float = 0.1) -> str | None:
    """Flags when a clinician might stop at the top diagnosis even though
    a commonly-confused alternative scored within `margin` of it —
    this is the concrete, checkable version of the pitch's
    'anchoring bias detected: consider alternative diagnoses' feature."""
    if len(results) < 2:
        return None
    top, runner_up = results[0], results[1]
    if (top.confidence - runner_up.confidence) <= margin:
        return (f"Anchoring bias risk: '{top.disease}' ({top.confidence:.0%}) and "
                f"'{runner_up.disease}' ({runner_up.confidence:.0%}) are close enough "
                f"that ruling out only the top result risks a missed diagnosis.")
    return None


def explain(results: list) -> str:
    lines = []
    for r in results:
        lines.append(f"- {r.disease}: {r.confidence:.0%} confidence "
                      f"(matched: {', '.join(r.matched_symptoms) or 'none'})")
        if r.missing_key_symptoms:
            lines.append(f"    missing hallmark signs that would raise confidence: "
                          f"{', '.join(r.missing_key_symptoms)}")
    return "\n".join(lines)
