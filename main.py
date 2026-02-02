import re
import string
import time
from collections import Counter

# --- NLTK ---
import nltk
from nltk.corpus import stopwords
nltk.download("punkt")
nltk.download("stopwords")

# --- TextBlob ---
from textblob import TextBlob


try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception as e:
    print("spaCy unavailable:", e)
    SPACY_AVAILABLE = False



def read_file():
    with open("alice29.txt", "r", encoding="utf-8") as f:
        return f.read()



def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n|\t|\s+", " ", text)

    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^\w\s]", "", text)   # remove emoticons

    stop = set(stopwords.words("english"))
    words = [w for w in text.split() if w not in stop]

    cleaned = " ".join(words)
    return cleaned, words



def tokenize_nltk(text):
    s = nltk.sent_tokenize(text)
    w = nltk.word_tokenize(text)
    return s, w


def tokenize_textblob(text):
    blob = TextBlob(text)
    s = [str(x) for x in blob.sentences]
    w = [str(x) for x in blob.words]
    return s, w


def tokenize_spacy(text):
    if not SPACY_AVAILABLE:
        return [], []
    doc = nlp(text)
    s = [sent.text for sent in doc.sents]
    w = [token.text for token in doc if token.is_alpha]
    return s, w



def top_10(words):
    count = Counter(words)
    return count.most_common(10)



def measure(func, text):
    start = time.time()
    func(text)
    return time.time() - start


def compare_times(text):
    results = {}

    # NLTK
    results["nltk_sentence"] = measure(lambda t: tokenize_nltk(t)[0], text)
    results["nltk_words"] = measure(lambda t: tokenize_nltk(t)[1], text)

    # TextBlob
    results["textblob_sentence"] = measure(lambda t: tokenize_textblob(t)[0], text)
    results["textblob_words"] = measure(lambda t: tokenize_textblob(t)[1], text)

    # spaCy
    if SPACY_AVAILABLE:
        results["spacy_sentence"] = measure(lambda t: tokenize_spacy(t)[0], text)
        results["spacy_words"] = measure(lambda t: tokenize_spacy(t)[1], text)
    else:
        results["spacy_sentence"] = "N/A"
        results["spacy_words"] = "N/A"

    return results


def main():
    text = read_file()

    cleaned, cleaned_words = clean_text(text)
    with open("cleaned.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)

    with open("words.txt", "w", encoding="utf-8") as f:
        for w in cleaned_words:
            f.write(w + "\n")

    top = top_10(cleaned_words)
    with open("top10words.txt", "w", encoding="utf-8") as f:
        for word, count in top:
            f.write(f"{word} {count}\n")

    results = compare_times(text)
    with open("time_compares.txt", "w", encoding="utf-8") as f:
        f.write("Framework Sentence_Time Word_Time\n")
        f.write(f"NLTK {results['nltk_sentence']} {results['nltk_words']}\n")
        f.write(f"TextBlob {results['textblob_sentence']} {results['textblob_words']}\n")
        f.write(f"spaCy {results['spacy_sentence']} {results['spacy_words']}\n")


if __name__ == "__main__":
    main()
