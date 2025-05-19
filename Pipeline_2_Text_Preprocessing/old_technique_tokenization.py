import re
import os
import sys
import logging
import pandas as pd
import spacy
from spacy.matcher import Matcher
from tokenizers import ByteLevelBPETokenizer
import contractions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# === Config ===
CSV_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Intent_Data_Acquisition/Day2_cleaned_dataset.csv"
TEXT_COL = "utterance"
CORPUS_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/2_Pipeline_Text_Preprocessing/corpus.txt"
SAVE_DIR = "bpe_tokenizer"
OUTPUT_CSV = "Day2_Tokenized_dataset.csv"

# === Precompiled regex patterns ===
EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}")
ACRONYM = re.compile(r"(?:[A-Za-z]\.){2,}|[A-Za-z](?:\.[A-Za-z])+")
NUM_UNIT = re.compile(r"\d+(?:\.\d+)?\s*(?:KM|M|cm|mm|kg|km/h|mph|%|\$)")
DECIMAL = re.compile(r"\d+\.\d+")

# Slang dictionary
SLANG_ABBREVIATIONS = {
    # 📱 Social Media & Messaging Platforms
    "insta": "instagram",
    "fb": "facebook",
    "wp": "whatsapp",
    "tg": "telegram",
    "yt": "youtube",
    "tw": "twitter",
    "sc": "snapchat",
    "tt": "tiktok",
    "li": "linkedin",
    "pin": "pinterest",
    "dc": "discord",

    # 💬 General Chat Abbreviations
    "e.g.": "example",
    "e.g": "example",
    "ex": "example",
    "u": "you",
    "ur": "your",
    "r": "are",
    "b4": "before",
    "gr8": "great",
    "l8r": "later",
    "bday": "birthday",
    "idk": "i do not know",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "brb": "be right back",
    "ttyl": "talk to you later",
    "omg": "oh my god",
    "lol": "laughing out loud",
    "btw": "by the way",
    "afaik": "as far as i know",
    "fyi": "for your information",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "irl": "in real life",
    "np": "no problem",
    "thx": "thanks",
    "ty": "thank you",
    "yw": "you are welcome",
    "wtf": "what the heck",  # softened
    "wth": "what the heck",
    "bc": "because",
    "bcz": "because",
    "cuz": "because",
    "plz": "please",
    "pls": "please",
    "omw": "on my way",
    "jk": "just kidding",
    "gg": "good game",
    "hf": "have fun",
    "np": "no problem",
    "wyd": "what are you doing",
    "hbu": "how about you",
    "ilu": "i love you",
    "ily": "i love you",
    "bff": "best friend forever"
}


def load_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
        logging.info(f"Loaded SpaCy model '{model_name}' successfully.")
        return nlp
    except OSError:
        logging.error(
            f"Failed to load SpaCy model '{model_name}'. Please run: python -m spacy download {model_name}"
        )
        sys.exit(1)

def build_phrasal_verb_matcher(nlp):
    matcher = Matcher(nlp.vocab)
    verbs = ["shut", "turn", "log"]
    preps = ["down", "off", "out"]
    patterns = [[{"LEMMA": v}, {"LEMMA": p}] for v in verbs for p in preps]
    matcher.add("PhrasalVerbs", patterns)
    return matcher

def smart_tokenize(text, nlp, matcher):
    text = text.strip().lower()
    text = contractions.fix(text)
    if not text:
        return []

    tokens = []
    doc = nlp(text)
    matches = matcher(doc)
    group_map = {start: end for _, start, end in matches}
    skip_until = -1

    for i, token in enumerate(doc):
        if i < skip_until:
            continue
        if i in group_map:
            span = doc[i:group_map[i]]
            # join without spaces so the BPE sees one token (e.g. 'shutdown')
            joined = "".join([tok.text for tok in span])
            tokens.append(joined)
            skip_until = group_map[i]
            continue

        txt = token.text
        if txt in SLANG_ABBREVIATIONS:
            tokens.extend(SLANG_ABBREVIATIONS[txt].split())
            continue
        if EMAIL.fullmatch(txt) or ACRONYM.fullmatch(txt) or \
           NUM_UNIT.fullmatch(txt) or DECIMAL.fullmatch(txt):
            tokens.append(txt)
            continue
        if token.pos_ in ("VERB", "AUX"):
            tokens.append(token.lemma_)
        else:
            tokens.append(txt)

    return tokens

def main():
    if not os.path.exists(CSV_PATH):
        logging.error(f"CSV file '{CSV_PATH}' not found.")
        sys.exit(1)
    try:
        df = pd.read_csv(
            CSV_PATH,
            dtype=str,
            sep='|',
            engine='python',
            on_bad_lines='skip',
            skip_blank_lines=True
        )
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        sys.exit(1)

    df.columns = [col.strip() for col in df.columns]
    if TEXT_COL not in df.columns:
        logging.error(f"Column '{TEXT_COL}' not found in CSV.")
        sys.exit(1)
    utterances = df[TEXT_COL].fillna("")

    nlp = load_spacy_model()
    matcher = build_phrasal_verb_matcher(nlp)

    logging.info(f"Tokenizing corpus and writing to {CORPUS_PATH}...")
    tokenized_cache = {}
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for line in utterances:
            tokens = smart_tokenize(line, nlp, matcher)
            tokenized_cache[line] = tokens
            f.write(" ".join(tokens) + "\n")

    tokenizer = ByteLevelBPETokenizer()
    try:
        tokenizer.train(
            files=[CORPUS_PATH],
            vocab_size=30000,
            min_frequency=2,
            special_tokens=["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"]
        )
        logging.info("Tokenizer training completed.")
    except Exception as e:
        logging.error(f"Tokenizer training failed: {e}")
        sys.exit(1)

    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer.save_model(SAVE_DIR, prefix="bpe")
    logging.info(f"Tokenizer files saved to '{SAVE_DIR}/'")

    logging.info(f"Encoding tokens and saving to {OUTPUT_CSV}...")
    def encode_text(text):
        tokens = tokenized_cache.get(text) or smart_tokenize(text, nlp, matcher)
        enc = tokenizer.encode(" ".join(tokens))
        return enc.tokens, enc.ids

    df["tokens"], df["token_ids"] = zip(*utterances.map(encode_text))
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"✅ Tokenized dataset saved as '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    main()
