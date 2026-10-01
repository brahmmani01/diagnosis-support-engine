"""
Generates synthetic patient vignettes for training and evaluating the
classifier. Each vignette is a noisy draw from a disease's symptom
profile: a symptom with weight w is present with probability w, plus
a small chance of an unrelated "noise" symptom, mimicking how real
patients rarely present with the textbook-perfect symptom set.

This is what lets diagnosis_classifier.py be a genuinely trained and
evaluated model instead of a hand-tuned lookup table: we can generate
as much labeled data as we want from the knowledge base, train on one
split, and measure accuracy on a held-out split the model never saw.
"""
import json
import random
from pathlib import Path

ALL_SYMPTOMS = [
    "fatigue", "joint_pain", "fever", "rash", "headache", "sleep_disturbance",
    "shortness_of_breath", "chest_pain", "rapid_heartrate", "anxiety_feeling",
    "cough", "chest_tightness", "tremor", "rigidity", "slow_movement",
    "balance_issues", "weight_gain", "cold_intolerance", "depressed_mood",
    "weight_loss", "hemoptysis",
]


def load_knowledge_base(path="knowledge_base.json"):
    with open(path) as f:
        return json.load(f)["diseases"]


def generate_vignette(disease: dict, noise_rate: float = 0.08, rng=random):
    """One synthetic patient: each hallmark symptom present with its own
    probability (not deterministically), plus a chance of 1-2 unrelated
    'noise' symptoms a real patient might also happen to report."""
    symptoms = set()
    for s, weight in disease["symptoms"].items():
        if rng.random() < weight:
            symptoms.add(s)
    for s in ALL_SYMPTOMS:
        if s not in disease["symptoms"] and rng.random() < noise_rate:
            symptoms.add(s)
    return sorted(symptoms)


def generate_dataset(knowledge_base, n_per_disease=200, seed=42):
    rng = random.Random(seed)
    rows = []
    for disease in knowledge_base:
        for _ in range(n_per_disease):
            symptoms = generate_vignette(disease, rng=rng)
            if symptoms:  # skip the rare empty-presentation draw
                rows.append({"symptoms": symptoms, "label": disease["name"]})
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    kb = load_knowledge_base()
    dataset = generate_dataset(kb, n_per_disease=200)
    with open("vignettes.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} synthetic vignettes across {len(kb)} diseases")
    print(f"Saved to vignettes.json")
