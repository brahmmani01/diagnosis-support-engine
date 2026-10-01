import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp_extractor import RegexSymptomExtractor
from vignette_generator import load_knowledge_base, generate_dataset
from diagnosis_classifier import DiagnosisClassifier, vectorize, evaluate
from fhir_adapter import parse_observation_bundle, build_diagnostic_report, SNOMED_MAP


def test_negation_is_excluded():
    e = RegexSymptomExtractor()
    result = e.extract("No fever, no rash. Patient has joint pain and fatigue.")
    assert "fever" not in result
    assert "rash" not in result
    assert "joint_pain" in result
    assert "fatigue" in result


def test_negation_scope_does_not_cross_but():
    e = RegexSymptomExtractor()
    result = e.extract("Denies chest pain but reports shortness of breath.")
    assert "chest_pain" not in result
    assert "shortness_of_breath" in result


def test_vectorize_shape():
    from vignette_generator import ALL_SYMPTOMS
    v = vectorize(["fatigue", "fever"])
    assert len(v) == len(ALL_SYMPTOMS)
    assert v.sum() == 2


def test_classifier_trains_and_predicts_known_pattern():
    kb = load_knowledge_base()
    dataset = generate_dataset(kb, n_per_disease=100, seed=1)
    X = [row["symptoms"] for row in dataset]
    y = [row["label"] for row in dataset]
    clf = DiagnosisClassifier().fit(X, y)
    # a textbook Parkinson's presentation should rank Parkinson's or
    # Essential Tremor in the top 2 (the known confusable pair)
    ranked = clf.predict_proba_ranked(["tremor", "rigidity", "slow_movement"], top_n=2)
    top_names = [name for name, _ in ranked]
    assert "Parkinson's Disease" in top_names


def test_fhir_roundtrip():
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"resourceType": "Observation",
                          "code": {"coding": [{"code": SNOMED_MAP["tremor"]}]}}},
        ],
    }
    symptoms = parse_observation_bundle(bundle)
    assert symptoms == ["tremor"]

    report = build_diagnostic_report([("Parkinson's Disease", 0.8)])
    assert report["resourceType"] == "DiagnosticReport"
    assert report["codedDiagnosis"][0]["coding"][0]["code"] == "G20"


def test_top3_accuracy_exceeds_top1_meaningfully():
    """Sanity check on the model's own evaluation: top-3 accuracy should
    be substantially higher than top-1, since many diseases here share
    overlapping symptom profiles by design (that's the whole point)."""
    kb = load_knowledge_base()
    dataset = generate_dataset(kb, n_per_disease=150, seed=2)
    X = [row["symptoms"] for row in dataset]
    y = [row["label"] for row in dataset]
    split = int(len(X) * 0.75)
    clf = DiagnosisClassifier().fit(X[:split], y[:split])
    result = evaluate(clf, X[split:], y[split:])
    assert result["top3_accuracy"] > result["top1_accuracy"]
    assert result["top3_accuracy"] > 0.80


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
