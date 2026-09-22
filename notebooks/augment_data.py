"""
ISL Gloss Data Augmentation Script.
Augments 101 sentence-gloss pairs to 500+ using:
1. Rule-based ISL grammar transforms
2. Template-based paraphrasing
"""

import csv
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "Dataset" / "data" / "isl_csltr" / "ISL_CSLRT_Corpus" / "ISL_CSLRT_Corpus" / "corpus_csv_files" / "ISL Corpus sign glosses.csv"
OUTPUT_PATH = PROJECT_DIR / "backend" / "augmented_glosses.csv"

# ISL grammar rules: drop these words from English
ISL_DROP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "am", "do", "does", "did", "can", "could", "would", "should", "may", "might"}

# Word order transformation: move subject to end (simplified SOV)
def apply_isl_grammar(sentence):
    words = sentence.lower().strip().split()
    filtered = [w for w in words if w.strip(".,!?;:'\"") not in ISL_DROP_WORDS]
    if not filtered:
        filtered = words
    return " ".join(filtered)


def to_gloss_format(isl_words):
    return " ".join(w.upper().strip(".,!?;:'\"") for w in isl_words.split() if w.strip(".,!?;:'\""))


# Template-based paraphrases
PARAPHRASE_TEMPLATES = {
    "greeting": [
        "hey how are you doing",
        "hello how have you been",
        "hi there how are things",
        "greetings how are you",
    ],
    "question": [
        "tell me about {topic}",
        "what do you think about {topic}",
        "explain {topic} to me",
        "can you describe {topic}",
    ],
    "request": [
        "please {action}",
        "could you {action}",
        "would you mind {action}",
        "I need you to {action}",
    ],
    "statement": [
        "I want to {action}",
        "I need to {action}",
        "I have to {action}",
        "let me {action}",
    ],
}

# Sentence pattern detection for augmentation
def detect_pattern(sentence):
    s = sentence.lower()
    if any(w in s for w in ["hello", "hi", "hey", "greet"]):
        return "greeting"
    if "?" in sentence or any(w in s for w in ["what", "how", "why", "where", "when", "who", "which"]):
        return "question"
    if any(w in s for w in ["please", "could you", "can you", "would you"]):
        return "request"
    return "statement"


# Rule-based augmentations
def augment_rule_based(sentence, glosses):
    augmented = []
    words = sentence.lower().strip().split()

    # 1. Drop articles/auxiliaries
    isl_form = apply_isl_grammar(sentence)
    if isl_form != sentence.lower().strip():
        augmented.append((isl_form, to_gloss_format(isl_form)))

    # 2. Contractions expansion/contraction
    contractions = {
        "don't": ["do not", "dont"],
        "can't": ["cannot", "can not"],
        "won't": ["will not"],
        "i'm": ["i am"],
        "you're": ["you are"],
        "he's": ["he is"],
        "she's": ["she is"],
        "it's": ["it is"],
        "we're": ["we are"],
        "they're": ["they are"],
        "i've": ["i have"],
        "you've": ["you have"],
        "we've": ["we have"],
        "they've": ["they have"],
        "isn't": ["is not"],
        "aren't": ["are not"],
        "wasn't": ["was not"],
        "weren't": ["were not"],
        "hasn't": ["has not"],
        "haven't": ["have not"],
        "hadn't": ["had not"],
    }

    for contraction, expansions in contractions.items():
        if contraction in sentence.lower():
            for exp in expansions[:1]:
                new_sent = sentence.lower().replace(contraction, exp)
                augmented.append((new_sent, to_gloss_format(apply_isl_grammar(new_sent))))
            break

    # 3. Add politeness markers
    if not any(w in sentence.lower() for w in ["please", "thank", "sorry"]):
        augmented.append((f"please {sentence.lower().strip()}", to_gloss_format(apply_isl_grammar(f"please {sentence.lower().strip()}"))))

    # 4. Formality variations
    if "you" in words:
        new_sent = sentence.lower().replace("you", "you sir")
        augmented.append((new_sent, to_gloss_format(apply_isl_grammar(new_sent))))

    return augmented


def load原始数据():
    pairs = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row["Sentence"].strip()
            glosses = row["SIGN GLOSSES"].strip()
            pairs.append((sentence, glosses))
    return pairs


def augment_dataset():
    original = load原始数据()
    augmented = list(original)

    for sentence, glosses in original:
        # Rule-based augmentations
        rule_aug = augment_rule_based(sentence, glosses)
        for new_sent, new_gloss in rule_aug:
            if new_sent != sentence.lower().strip() and (new_sent, new_gloss) not in augmented:
                augmented.append((new_sent, new_gloss))

        # Template paraphrases based on pattern
        pattern = detect_pattern(sentence)
        if pattern in PARAPHRASE_TEMPLATES:
            for template in random.sample(PARAPHRASE_TEMPLATES[pattern], min(2, len(PARAPHRASE_TEMPLATES[pattern]))):
                if "{topic}" in template:
                    topic_words = [w for w in sentence.lower().split() if w not in ISL_DROP_WORDS and len(w) > 2]
                    if topic_words:
                        topic = " ".join(topic_words[:2])
                        new_sent = template.format(topic=topic)
                        augmented.append((new_sent, to_gloss_format(apply_isl_grammar(new_sent))))
                elif "{action}" in template:
                    action_words = [w for w in sentence.lower().split() if w not in ISL_DROP_WORDS and len(w) > 2]
                    if action_words:
                        action = " ".join(action_words[:3])
                        new_sent = template.format(action=action)
                        augmented.append((new_sent, to_gloss_format(apply_isl_grammar(new_sent))))

    # Deduplicate
    seen = set()
    unique = []
    for sent, gloss in augmented:
        key = sent.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append((sent, gloss))

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "output"])
        for sent, gloss in unique:
            writer.writerow([sent, gloss])

    print(f"Original pairs: {len(original)}")
    print(f"Augmented pairs: {len(unique)}")
    print(f"Output: {OUTPUT_PATH}")
    return unique


if __name__ == "__main__":
    random.seed(42)
    augment_dataset()
