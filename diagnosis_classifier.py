"""
A genuinely trained, evaluated differential-diagnosis classifier —
the v2 upgrade from the rule-based weighted-overlap scorer in
diagnosis_engine.py.

Model: multinomial logistic regression over a binary symptom-presence
feature vector (one-vs-rest). Chosen over a black-box model
deliberately: every prediction's coefficients are directly readable,
so "why did it suggest this" has a real answer (see explain.py) —
the same explainability requirement a clinical decision-support tool
has to satisfy, and the same quality Lumeris's own public materials
describe wanting from "augment, don't replace" AI.

Trained and evaluated on synthetic vignettes (vignette_generator.py)
generated FROM the knowledge base's own symptom-weight profiles, so
this measures how well the model recovers known clinical patterns
from noisy presentations — not real-world generalization. That
distinction matters and is stated plainly in the README; a model
intended for real deployment would be trained on de-identified EHR
data or MIMIC-III/i2b2 corpora instead.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from vignette_generator import ALL_SYMPTOMS, load_knowledge_base, generate_dataset


def vectorize(symptom_list, vocab=ALL_SYMPTOMS):
    v = np.zeros(len(vocab))
    for s in symptom_list:
        if s in vocab:
            v[vocab.index(s)] = 1
    return v


class DiagnosisClassifier:
    def __init__(self, vocab=ALL_SYMPTOMS):
        self.vocab = vocab
        self.model = LogisticRegression(max_iter=1000)
        self.label_encoder = LabelEncoder()

    def fit(self, X_symptoms, y_labels):
        X = np.array([vectorize(s, self.vocab) for s in X_symptoms])
        y = self.label_encoder.fit_transform(y_labels)
        self.model.fit(X, y)
        return self

    def predict_proba_ranked(self, symptom_list, top_n=5):
        x = vectorize(symptom_list, self.vocab).reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        order = np.argsort(probs)[::-1][:top_n]
        return [(self.label_encoder.inverse_transform([i])[0], float(probs[i])) for i in order]

    def top_features(self, disease_name, n=5):
        """Which symptoms most increase the model's confidence in this
        disease - real feature attribution, not a hand-picked list."""
        class_idx = list(self.label_encoder.classes_).index(disease_name)
        coefs = self.model.coef_[class_idx]
        order = np.argsort(coefs)[::-1][:n]
        return [(self.vocab[i], float(coefs[i])) for i in order]


def evaluate(clf, X_test, y_test):
    X = np.array([vectorize(s, clf.vocab) for s in X_test])
    y_true = clf.label_encoder.transform(y_test)
    y_pred = clf.model.predict(X)
    probs = clf.model.predict_proba(X)

    top1_acc = accuracy_score(y_true, y_pred)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    top3_acc = np.mean([y_true[i] in top3[i] for i in range(len(y_true))])

    return {
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "classes": clf.label_encoder.classes_,
    }


def confused_pair_accuracy(clf, kb, X_test_syms, y_test, result):
    """Accuracy specifically on the commonly-confused pairs named in the
    knowledge base - the exact misdiagnosis scenarios this tool targets.
    This is the number that matters most for the project's stated goal."""
    confused_names = set()
    name_to_disease = {d["name"]: d for d in kb}
    for d in kb:
        for other in d.get("commonly_confused_with", []):
            confused_names.add(d["name"])
            confused_names.add(other)

    mask = [label in confused_names for label in y_test]
    if not any(mask):
        return None
    idx = [i for i, m in enumerate(mask) if m]
    y_true_sub = result["y_true"][idx]
    y_pred_sub = result["y_pred"][idx]
    return accuracy_score(y_true_sub, y_pred_sub)


if __name__ == "__main__":
    kb = load_knowledge_base()
    dataset = generate_dataset(kb, n_per_disease=200)

    X = [row["symptoms"] for row in dataset]
    y = [row["label"] for row in dataset]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = DiagnosisClassifier().fit(X_train, y_train)
    result = evaluate(clf, X_test, y_test)

    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    print(f"Top-1 accuracy: {result['top1_accuracy']:.1%}")
    print(f"Top-3 accuracy: {result['top3_accuracy']:.1%}")

    confused_acc = confused_pair_accuracy(clf, kb, X_test, y_test, result)
    print(f"Accuracy on commonly-confused disease pairs specifically: {confused_acc:.1%}")

    print("\n--- Per-disease performance ---")
    print(classification_report(result["y_true"], result["y_pred"],
                                 target_names=result["classes"], zero_division=0))

    print("--- Example: what drives a 'Hypothyroidism' prediction ---")
    for symptom, weight in clf.top_features("Hypothyroidism"):
        print(f"  {symptom}: {weight:+.2f}")

    import pickle
    with open("trained_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("\nSaved trained_classifier.pkl")
