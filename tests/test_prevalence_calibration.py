import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prevalence_calibration import recalibrate


def test_recalibration_shifts_toward_more_common_disease():
    raw = [("Pulmonary Embolism", 0.94), ("Anxiety Disorder", 0.047),
           ("Asthma", 0.005), ("COPD", 0.004), ("Pneumonia", 0.002)]
    priors = {"Pulmonary Embolism": 0.10, "Anxiety Disorder": 0.50,
              "Asthma": 0.25, "COPD": 0.10, "Pneumonia": 0.05}
    calibrated = recalibrate(raw, priors)
    calibrated_dict = dict(calibrated)
    # Anxiety's calibrated share should rise relative to its raw share,
    # since its real-world prior is far higher than the implicit
    # uniform prior the raw model assumed
    raw_dict = dict(raw)
    assert calibrated_dict["Anxiety Disorder"] > raw_dict["Anxiety Disorder"]
    # PE should still rank first (symptoms are genuinely more
    # PE-specific), just with a less overconfident number
    assert calibrated[0][0] == "Pulmonary Embolism"
    assert calibrated_dict["Pulmonary Embolism"] < raw_dict["Pulmonary Embolism"]


def test_recalibration_still_sums_to_one():
    raw = [("A", 0.7), ("B", 0.2), ("C", 0.1)]
    priors = {"A": 0.3, "B": 0.3, "C": 0.4}
    calibrated = recalibrate(raw, priors)
    assert abs(sum(c for _, c in calibrated) - 1.0) < 1e-9


def test_disease_without_prior_is_left_unscaled_relative_to_others():
    raw = [("A", 0.9), ("Unlisted", 0.1)]
    priors = {"A": 0.5}
    calibrated = recalibrate(raw, priors)
    # "Unlisted" keeps its raw value before renormalization - just
    # confirm it doesn't error and still appears in the output
    names = [n for n, _ in calibrated]
    assert "Unlisted" in names


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
