import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diagnosis_engine import rank_differentials, detect_anchoring_bias, score_disease, load_knowledge_base


def test_top_match_is_correct_disease():
    results = rank_differentials(["tremor", "balance_issues"])
    assert results[0].disease == "Essential Tremor"


def test_confidence_is_bounded():
    kb = load_knowledge_base(str(Path(__file__).parent.parent / "knowledge_base.json"))
    for disease in kb:
        r = score_disease(set(disease["symptoms"].keys()), disease)
        assert 0.0 <= r.confidence <= 1.0

def test_full_symptom_match_is_high_confidence():
    kb = load_knowledge_base(str(Path(__file__).parent.parent / "knowledge_base.json"))
    for disease in kb:
        r = score_disease(set(disease["symptoms"].keys()), disease)
        assert r.confidence == 1.0


def test_no_matching_symptoms_returns_empty():
    results = rank_differentials(["unrelated_symptom_xyz"])
    assert results == []


def test_anchoring_bias_flagged_when_close():
    results = rank_differentials(["fatigue", "joint_pain", "headache"])
    flag = detect_anchoring_bias(results)
    assert flag is not None
    assert "Anchoring bias" in flag


def test_anchoring_bias_not_flagged_when_clear_winner():
    results = rank_differentials(["tremor", "balance_issues"])
    flag = detect_anchoring_bias(results)
    assert flag is None


if __name__ == "__main__":
    import inspect
    tests = [f for name, f in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
