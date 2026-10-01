"""
Runs the five commonly-misdiagnosed scenarios named in the course
project pitch deck through the differential-diagnosis engine, showing
ranked differentials, explanations, and the anchoring-bias flag.
"""
from diagnosis_engine import rank_differentials, detect_anchoring_bias, explain

CASES = {
    "Case 1 - fatigue + joint pain (Lyme vs. Chronic Fatigue/Fibromyalgia)":
        ["fatigue", "joint_pain", "headache"],
    "Case 2 - shortness of breath + chest pain (PE vs. Anxiety/Asthma)":
        ["shortness_of_breath", "chest_pain", "rapid_heartrate"],
    "Case 3 - tremor (Parkinson's vs. Essential Tremor)":
        ["tremor", "balance_issues"],
    "Case 4 - fatigue + weight gain (Hypothyroidism vs. Depression)":
        ["fatigue", "weight_gain", "depressed_mood"],
    "Case 5 - cough + chest pain (Lung Cancer vs. COPD/Pneumonia)":
        ["cough", "chest_pain", "shortness_of_breath", "weight_loss"],
}

if __name__ == "__main__":
    for title, symptoms in CASES.items():
        print(f"\n{'=' * 70}\n{title}")
        print(f"Reported symptoms: {', '.join(symptoms)}")
        results = rank_differentials(symptoms)
        print("\nRanked differential diagnoses:")
        print(explain(results))
        flag = detect_anchoring_bias(results)
        if flag:
            print(f"\n[BIAS ALERT] {flag}")
