# Tokenization, stopword removal, lemmatization

# src/preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs, mentions, hashtags, numbers
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|\d+", "", text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords + Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)
