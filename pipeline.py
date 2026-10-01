"""
The full v2 pipeline, wired end to end:

  clinical note (free text)
      -> RegexSymptomExtractor (or ClinicalBERTSymptomExtractor)
      -> DiagnosisClassifier (trained, evaluated logistic regression)
      -> anchoring-bias check (from the original v1 engine)
      -> FHIR DiagnosticReport

Run directly for a demo against the same 5 misdiagnosis-pair cases
used in the original v1 demo.py, but now starting from a free-text
note instead of a pre-selected symptom list.
"""
import pickle
from nlp_extractor import RegexSymptomExtractor
from fhir_adapter import build_diagnostic_report
from diagnosis_engine import detect_anchoring_bias
from diagnosis_classifier import DiagnosisClassifier  # noqa: F401 (needed for unpickling)

with open("trained_classifier.pkl", "rb") as f:
    CLASSIFIER = pickle.load(f)

EXTRACTOR = RegexSymptomExtractor()


def diagnose_from_note(note_text: str, top_n: int = 5, patient_reference="Patient/example"):
    symptoms = EXTRACTOR.extract(note_text)
    ranked = CLASSIFIER.predict_proba_ranked(symptoms, top_n=top_n)

    # reuse the v1 engine's bias-flag logic (expects (name, confidence)
    # objects with .disease/.confidence; adapt the tuple shape here)
    class _R:
        def __init__(self, disease, confidence):
            self.disease, self.confidence = disease, confidence
    bias_flag = detect_anchoring_bias([_R(n, c) for n, c in ranked])

    report = build_diagnostic_report(ranked, patient_reference=patient_reference)
    return {
        "extracted_symptoms": symptoms,
        "ranked_differential": ranked,
        "anchoring_bias_flag": bias_flag,
        "fhir_report": report,
    }


NOTES = {
    "Case 1 - Lyme vs Chronic Fatigue/Fibromyalgia":
        "Patient reports fatigue for several weeks, intermittent joint pain, "
        "and occasional headaches. No fever currently.",
    "Case 2 - PE vs Anxiety/Asthma":
        "Patient c/o shortness of breath and chest pain, reports palpitations. "
        "Denies recent travel.",
    "Case 3 - Parkinson's vs Essential Tremor":
        "Hand tremor noted, patient reports some balance issues when walking.",
    "Case 4 - Hypothyroidism vs Depression":
        "Patient reports fatigue, weight gain, and low mood over several months.",
    "Case 5 - Lung Cancer vs COPD/Pneumonia":
        "Persistent cough and chest pain, unintentional weight loss, "
        "some shortness of breath.",
}

if __name__ == "__main__":
    for title, note in NOTES.items():
        print(f"\n{'=' * 70}\n{title}")
        print(f"Note: {note}")
        result = diagnose_from_note(note)
        print(f"Extracted symptoms: {result['extracted_symptoms']}")
        print("Ranked differential:")
        for name, conf in result["ranked_differential"][:3]:
            print(f"  {name}: {conf:.0%}")
        if result["anchoring_bias_flag"]:
            print(f"[BIAS ALERT] {result['anchoring_bias_flag']}")
