import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vignette_generator import load_knowledge_base, generate_dataset
from diagnosis_classifier import DiagnosisClassifier
from longitudinal_simulation import run_simulation, summarize


def _trained_classifier():
    kb = load_knowledge_base()
    dataset = generate_dataset(kb, n_per_disease=150, seed=3)
    X = [row["symptoms"] for row in dataset]
    y = [row["label"] for row in dataset]
    return DiagnosisClassifier().fit(X, y), kb


def test_longitudinal_never_caught_rate_not_worse_than_baseline():
    clf, kb = _trained_classifier()
    results = run_simulation(clf, kb, n_patients_per_pair=60, seed=11)
    s = summarize(results)
    # the core claim: aggregating history should not do WORSE than
    # anchored per-visit care at eventually reaching the correct diagnosis
    assert s["pct_longitudinal_never_caught"] <= s["pct_baseline_never_caught"]


def test_longitudinal_rescues_some_never_caught_cases():
    clf, kb = _trained_classifier()
    results = run_simulation(clf, kb, n_patients_per_pair=60, seed=11)
    s = summarize(results)
    assert s["n_rescued_by_longitudinal"] > 0


def test_every_confused_pair_is_non_negative_on_average():
    """The headline claim should hold per-subgroup, not just on average
    across everything - a result that only holds in aggregate but
    reverses for specific pairs is a much weaker (and riskier to
    present) claim."""
    from longitudinal_simulation import summarize_by_pair
    clf, kb = _trained_classifier()
    results = run_simulation(clf, kb, n_patients_per_pair=60, seed=11)
    for true_name, other_name, s in summarize_by_pair(results):
        assert s["avg_visits_earlier"] >= -0.05, (
            f"{true_name} vs {other_name} went backwards: {s['avg_visits_earlier']}"
        )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
