"""
FHIR-shaped interface: accepts input the way an EMR would actually
send it (a Bundle of Observation resources, or a clinical note as a
DocumentReference/Composition-style string), and returns output as a
FHIR-like DiagnosticReport with ranked conditions and ICD-10 codes.

This is the piece that turns "a Python script with a demo.py" into
"something that could sit behind an API a health system's EMR calls" -
the actual integration shape, not just the model.

Not a full FHIR validator/library (that's a much bigger dependency);
this implements the specific resource shapes needed for this
engine's input/output contract, matching the real FHIR R4 structure
closely enough that adapting to a real `fhir.resources` Python
library later is a mechanical step, not a redesign.
"""
import json
from vignette_generator import load_knowledge_base
from diagnosis_classifier import DiagnosisClassifier  # noqa: F401 (needed for unpickling)

# Map our internal snake_case symptom codes to SNOMED CT codes, the
# standard EMRs actually use for Observation.code - a real integration
# would receive these, not our internal names.
SNOMED_MAP = {
    "fatigue": "84229001", "joint_pain": "57676002", "fever": "386661006",
    "rash": "271807003", "headache": "25064002", "sleep_disturbance": "193462001",
    "shortness_of_breath": "267036007", "chest_pain": "29857009",
    "rapid_heartrate": "3424008", "anxiety_feeling": "48694002", "cough": "49727002",
    "chest_tightness": "23924001", "tremor": "26079004", "rigidity": "233640000",
    "slow_movement": "16228001", "balance_issues": "282120003", "weight_gain": "8943002",
    "cold_intolerance": "287095007", "depressed_mood": "35489007",
    "weight_loss": "89362005", "hemoptysis": "66857006",
}
SNOMED_TO_SYMPTOM = {v: k for k, v in SNOMED_MAP.items()}


def parse_observation_bundle(bundle: dict) -> list:
    """Input: a FHIR Bundle of Observation resources (what an EMR would
    actually send - e.g. a set of patient-reported-symptom entries
    logged during intake). Output: our internal symptom code list."""
    symptoms = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Observation":
            continue
        for coding in resource.get("code", {}).get("coding", []):
            code = coding.get("code")
            if code in SNOMED_TO_SYMPTOM:
                symptoms.append(SNOMED_TO_SYMPTOM[code])
    return symptoms


def build_diagnostic_report(ranked_results, patient_reference="Patient/example", kb=None):
    """Output: a FHIR-like DiagnosticReport bundling the ranked
    differential as `conclusion` text plus structured `codedDiagnosis`
    entries with ICD-10 codes - the shape a clinician's EMR view or a
    downstream system could actually consume."""
    kb = kb or {d["name"]: d for d in load_knowledge_base()}
    coded = []
    for name, confidence in ranked_results:
        icd10 = kb.get(name, {}).get("icd10", "UNKNOWN")
        coded.append({
            "coding": [{"system": "http://hl7.org/fhir/sid/icd-10-cm",
                        "code": icd10, "display": name}],
            "text": f"{name} (model confidence: {confidence:.0%})",
        })
    return {
        "resourceType": "DiagnosticReport",
        "status": "preliminary",
        "subject": {"reference": patient_reference},
        "conclusion": (
            "AI-generated differential diagnosis suggestions for clinician "
            "review. Not a final diagnosis - confirm with clinical judgment "
            "and additional workup as indicated."
        ),
        "codedDiagnosis": coded,
    }


if __name__ == "__main__":
    example_bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Observation",
                          "code": {"coding": [{"system": "http://snomed.info/sct",
                                                "code": SNOMED_MAP["fatigue"]}]}}},
            {"resource": {"resourceType": "Observation",
                          "code": {"coding": [{"system": "http://snomed.info/sct",
                                                "code": SNOMED_MAP["weight_gain"]}]}}},
            {"resource": {"resourceType": "Observation",
                          "code": {"coding": [{"system": "http://snomed.info/sct",
                                                "code": SNOMED_MAP["cold_intolerance"]}]}}},
        ],
    }
    symptoms = parse_observation_bundle(example_bundle)
    print("Parsed symptoms from FHIR Bundle:", symptoms)

    import pickle
    with open("trained_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    ranked = clf.predict_proba_ranked(symptoms, top_n=3)
    report = build_diagnostic_report(ranked, patient_reference="Patient/12345")
    print("\nFHIR DiagnosticReport:")
    print(json.dumps(report, indent=2))
