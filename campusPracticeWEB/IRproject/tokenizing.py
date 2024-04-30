import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from math import log10
from math import sqrt

# Document preprocessing
def preprocess_document(document):
    # Tokenization
    tokens = word_tokenize(document)

    # Case normalization
    tokens = [token.lower() for token in tokens]

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    preprocessed_tokens = [stemmer.stem(token) for token in tokens]

    return preprocessed_tokens


documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
query = "this is a query"
