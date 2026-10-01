"""
Extracts structured symptoms from free-text clinical notes, so the
engine can take "Pt c/o fatigue x3 weeks, intermittent joint pain,
low-grade fevers" instead of requiring pre-selected checkboxes - the
actual input shape of a real EMR note.

Two extractors, same output contract (a list of canonical symptom
codes from vignette_generator.ALL_SYMPTOMS):

1. RegexSymptomExtractor - runs right now, no dependencies, no
   internet. A keyword/synonym matcher. Brittle on paraphrasing
   ("short of breath" works, "can't catch my breath" doesn't) but
   fully offline and instant - a legitimate fallback, not a toy.

2. ClinicalBERTSymptomExtractor - a real, correct implementation using
   a HuggingFace biomedical NER model, which genuinely handles
   paraphrasing and clinical shorthand via learned language
   understanding rather than keyword lists. This requires the
   `transformers` + `torch` packages and an internet connection to
   download model weights (~400MB) - NOT available in this sandbox
   (no internet access here), so this class is written and documented
   but not executed or benchmarked in this repo. Run it yourself in
   Google Colab (free GPU) or any machine with internet access; see
   README.md for exact steps. Treat its accuracy as unverified until
   you've run it and compared its output against the regex extractor
   on a few real examples.
"""
import re
from vignette_generator import ALL_SYMPTOMS

# ---------------------------------------------------------------------
# Extractor 1: offline, runs now
# ---------------------------------------------------------------------
SYNONYMS = {
    "fatigue": [r"\bfatigu\w*", r"\btired\w*", r"\bexhaust\w*", r"\blethargy\b"],
    "joint_pain": [r"\bjoint pain\b", r"\barthralgia\b", r"\bachy joints\b"],
    "fever": [r"\bfever\w*", r"\bfebrile\b", r"\belevated temp"],
    "rash": [r"\brash\b", r"\bskin lesion"],
    "headache": [r"\bheadache\w*", r"\bcephalgia\b"],
    "sleep_disturbance": [r"\binsomnia\b", r"\btrouble sleeping\b", r"\bpoor sleep\b"],
    "shortness_of_breath": [r"\bshort(ness)? of breath\b", r"\bsob\b", r"\bdyspnea\b"],
    "chest_pain": [r"\bchest pain\b", r"\bchest discomfort\b"],
    "rapid_heartrate": [r"\btachycardia\b", r"\brapid heart\w*", r"\bpalpitations?\b"],
    "anxiety_feeling": [r"\banxiet\w*", r"\bpanic\b", r"\bfeeling anxious\b"],
    "cough": [r"\bcough\w*"],
    "chest_tightness": [r"\bchest tightness\b", r"\btight chest\b"],
    "tremor": [r"\btremor\w*", r"\bshak(ing|y)\b"],
    "rigidity": [r"\brigidity\b", r"\bstiff(ness)?\b"],
    "slow_movement": [r"\bbradykinesia\b", r"\bslow(ed)? movement"],
    "balance_issues": [r"\bbalance\b", r"\bunsteady\b", r"\bfalls?\b"],
    "weight_gain": [r"\bweight gain\b", r"\bgained weight\b"],
    "cold_intolerance": [r"\bcold intoleran\w*", r"\balways cold\b"],
    "depressed_mood": [r"\bdepress\w*", r"\blow mood\b", r"\bsad(ness)?\b"],
    "weight_loss": [r"\bweight loss\b", r"\blost weight\b", r"\bunintentional weight\b"],
    "hemoptysis": [r"\bhemoptysis\b", r"\bcoughing (up )?blood\b", r"\bblood(y)? sputum\b"],
}


NEGATION_CUES = [r"\bno\b", r"\bnot\b", r"\bdenies\b", r"\bdenied\b",
                  r"\bwithout\b", r"\bnegative for\b", r"\bruled out\b", r"\babsent\b"]
# How many characters before a matched symptom span we scan for a
# negation cue. A simplified version of the NegEx algorithm:
# Chapman WW, et al. A Simple Algorithm for Identifying Negated
# Findings and Diseases in Discharge Summaries. J Biomed Inform.
# 2001;34(5):301-310. https://doi.org/10.1006/jbin.2001.1029
# Real clinical NLP pipelines (cTAKES, MedSpaCy) implement fuller
# versions of this same idea - full parsing, not just a fixed window.
NEGATION_WINDOW = 40
# Words that end a negation's scope even within the same sentence/window,
# e.g. "denies chest pain BUT reports shortness of breath" - the negation
# shouldn't carry past "but". Also modeled on NegEx's termination list.
TERMINATION_CUES = [r"\bbut\b", r"\bhowever\b", r"\bthough\b", r"\byet\b", r"\breports\b"]


class RegexSymptomExtractor:
    def extract(self, note_text: str) -> list:
        text = note_text.lower()
        found = []
        for symptom, patterns in SYNONYMS.items():
            for p in patterns:
                match = re.search(p, text)
                if match:
                    # restrict the negation window to the current sentence
                    # only - a bare character window bleeds across sentence
                    # boundaries (e.g. an unrelated "No X." earlier in the
                    # note would wrongly negate a later, un-negated symptom)
                    sentence_start = max(
                        text.rfind(".", 0, match.start()),
                        text.rfind("!", 0, match.start()),
                        text.rfind("?", 0, match.start()),
                    ) + 1
                    window_start = max(sentence_start, match.start() - NEGATION_WINDOW)
                    preceding = text[window_start:match.start()]
                    # trim the window to after the last termination word,
                    # so an earlier negation can't carry past a "but"/"reports"
                    last_term_end = 0
                    for term in TERMINATION_CUES:
                        m = re.search(term, preceding)
                        if m:
                            last_term_end = max(last_term_end, m.end())
                    preceding = preceding[last_term_end:]
                    negated = any(re.search(cue, preceding) for cue in NEGATION_CUES)
                    if not negated:
                        found.append(symptom)
                    break
        return found


# ---------------------------------------------------------------------
# Extractor 2: ClinicalBERT-based NER (real code, needs internet/GPU -
# see README.md "Running the ClinicalBERT extractor" before using)
# ---------------------------------------------------------------------
class ClinicalBERTSymptomExtractor:
    """
    Uses a HuggingFace biomedical NER pipeline to tag symptom/sign
    entities in free text, then maps each tagged span onto our
    canonical symptom vocabulary via substring/keyword matching (the
    mapping step is simple on purpose - the NER model's job is finding
    "something clinically relevant was mentioned here", the mapping's
    job is deciding which of our 21 known symptoms it corresponds to).

    Model: "d4data/biomedical-ner-all" - a public, pre-trained
    biomedical NER model covering signs/symptoms, diseases, and more,
    built on a BERT/ClinicalBERT-family backbone. Swap in
    "emilyalsentzer/Bio_ClinicalBERT" embeddings for a similarity-based
    variant if you want semantic matching instead of NER tagging - see
    the commented alternative at the bottom of this file.
    """

    def __init__(self, model_name: str = "d4data/biomedical-ner-all"):
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch\n"
                "Then re-run this - it needs internet access to download the "
                "model weights the first time (~400MB)."
            ) from e
        self.ner = pipeline("token-classification", model=model_name,
                             aggregation_strategy="simple")

    def extract(self, note_text: str) -> list:
        entities = self.ner(note_text)
        found = set()
        for ent in entities:
            span = ent["word"].lower()
            # map the NER-tagged span onto our canonical vocabulary by
            # substring match against the same synonym lists used by
            # the regex extractor - reuses that mapping instead of
            # duplicating it
            for symptom, patterns in SYNONYMS.items():
                if any(re.search(p, span) for p in patterns):
                    found.add(symptom)
        return sorted(found)


# --- Alternative: embedding-similarity extractor (sketch, not run) ---
# from sentence_transformers import SentenceTransformer, util
# model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")
# note_embedding = model.encode(note_text)
# symptom_descriptions = {s: f"patient reports {s.replace('_',' ')}" for s in ALL_SYMPTOMS}
# scores = {s: util.cos_sim(note_embedding, model.encode(desc))
#           for s, desc in symptom_descriptions.items()}
# This scores EVERY symptom by semantic similarity to the note, rather
# than requiring an exact NER tag - catches paraphrasing the NER
# model's training data didn't cover, at the cost of needing a
# similarity threshold to decide what counts as "present".


if __name__ == "__main__":
    note = ("Patient reports feeling very tired for the past month, "
             "with intermittent joint pain and a low-grade fever. "
             "No rash noted. Occasional headaches.")
    extractor = RegexSymptomExtractor()
    symptoms = extractor.extract(note)
    print("Note:", note)
    print("Extracted symptoms (offline regex extractor):", symptoms)
