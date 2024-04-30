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


# Indexing


class InvertedIndex:
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.document_frequencies = defaultdict(int)
        self.document_length = []

    def preprocess_document(self, document):
        tokens = word_tokenize(document.lower())
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))

        preprocessed_tokens = []
        for token in tokens:
            if token not in stop_words and token.isalpha():
                preprocessed_tokens.append(stemmer.stem(token))

        return preprocessed_tokens

    def add_document(self, document, doc_id):
        preprocessed_tokens = self.preprocess_document(document)
        term_frequencies = defaultdict(int)

        for term in preprocessed_tokens:
            term_frequencies[term] += 1

        for term, freq in term_frequencies.items():
            self.inverted_index[term].append((doc_id, freq))
            self.document_frequencies[term] += 1

        self.document_length.append(len(preprocessed_tokens))

    def build_index(self, documents):
        for doc_id, document in enumerate(documents):
            self.add_document(document, doc_id)

    def get_postings(self, term):
        return self.inverted_index.get(term, [])

    def get_document_frequency(self, term):
        return self.document_frequencies.get(term, 0)

    def get_document_length(self, doc_id):
        return self.document_length[doc_id]


# Query processing


class QueryProcessor:
    def __init__(self, inverted_index, document_lengths):
        self.inverted_index = inverted_index
        self.document_lengths = document_lengths

    def preprocess_query(self, query):
        tokens = word_tokenize(query.lower())
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))

        preprocessed_tokens = []
        for token in tokens:
            if token not in stop_words and token.isalpha():
                preprocessed_tokens.append(stemmer.stem(token))

        return preprocessed_tokens

    def process_query(self, query):
        preprocessed_query = self.preprocess_query(query)
        relevant_docs = set()

        for term in preprocessed_query:
            if term in self.inverted_index:
                relevant_docs.update(
                    [doc_id for doc_id, _ in self.inverted_index[term]]
                )

        ranked_docs = self.rank_documents(relevant_docs, preprocessed_query)
        return ranked_docs

    def rank_documents(self, relevant_docs, query_tokens):
        scores = {}

        for doc_id in relevant_docs:
            score = self.calculate_similarity(query_tokens, doc_id)
            scores[doc_id] = score

        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs

    def calculate_similarity(self, query_tokens, doc_id):
        score = 0
        document_length = self.document_lengths[doc_id]

        for term in query_tokens:
            if term in self.inverted_index:
                tf = 0
                for doc, freq in self.inverted_index[term]:
                    if doc == doc_id:
                        tf = freq
                        break
                cf = sum(freq for _, freq in self.inverted_index[term])
                df = len(self.inverted_index[term])
                tf_idf = (tf / document_length) * (cf / df)
                score += tf_idf

        return score


# Ranking
class RankDocuments:
    def __init__(self, documents):
        self.documents = documents
        self.inverted_index = defaultdict(dict)
        self.document_frequencies = defaultdict(int)
        self.document_lengths = {}
        self.total_documents = len(documents)
        self.avg_document_length = 0

        self.preprocess_documents()

    def preprocess_documents(self):
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))

        for doc_id, document in enumerate(self.documents):
            tokens = word_tokenize(document.lower())

            preprocessed_tokens = []
            term_frequencies = defaultdict(int)

            for token in tokens:
                if token not in stop_words and token.isalpha():
                    stemmed_token = stemmer.stem(token)
                    preprocessed_tokens.append(stemmed_token)
                    term_frequencies[stemmed_token] += 1

            self.document_lengths[doc_id] = sum(term_frequencies.values())

            for term, freq in term_frequencies.items():
                self.inverted_index[term][doc_id] = freq
                self.document_frequencies[term] += 1

        self.avg_document_length = (
            sum(self.document_lengths.values()) / self.total_documents
        )

    def rank(self, query):
        query_terms = self.preprocess_query(query)
        scores = defaultdict(float)

        for term in query_terms:
            if term in self.inverted_index:
                for doc_id, term_freq in self.inverted_index[term].items():
                    tf = term_freq / self.document_lengths[doc_id]
                    idf = log10(self.total_documents / self.document_frequencies[term])
                    scores[doc_id] += tf * idf

        ranked_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_documents

    def preprocess_query(self, query):
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        tokens = word_tokenize(query.lower())

        preprocessed_tokens = []
        for token in tokens:
            if token not in stop_words and token.isalpha():
                stemmed_token = stemmer.stem(token)
                preprocessed_tokens.append(stemmed_token)

        return preprocessed_tokens


# Similarity calculation (TF-IDF)
class CalculateSimilarity:
    def __init__(self, documents):
        self.documents = documents
        self.document_vectors = []
        self.query_vector = None

        self.preprocess_documents()

    def preprocess_documents(self):
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))

        for document in self.documents:
            tokens = word_tokenize(document.lower())

            preprocessed_tokens = []
            term_frequencies = defaultdict(int)

            for token in tokens:
                if token not in stop_words and token.isalpha():
                    stemmed_token = stemmer.stem(token)
                    preprocessed_tokens.append(stemmed_token)
                    term_frequencies[stemmed_token] += 1

            self.document_vectors.append(term_frequencies)

    def preprocess_query(self, query):
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        tokens = word_tokenize(query.lower())

        preprocessed_tokens = []
        term_frequencies = defaultdict(int)

        for token in tokens:
            if token not in stop_words and token.isalpha():
                stemmed_token = stemmer.stem(token)
                preprocessed_tokens.append(stemmed_token)
                term_frequencies[stemmed_token] += 1

        self.query_vector = term_frequencies

    def calculate_similarity(self):
        document_scores = []

        for document_vector in self.document_vectors:
            dot_product = 0
            doc_magnitude = 0

            for term, freq in self.query_vector.items():
                dot_product += freq * document_vector[term]

            for freq in document_vector.values():
                doc_magnitude += freq**2

            doc_magnitude = sqrt(doc_magnitude)
            similarity_score = dot_product / doc_magnitude
            document_scores.append(similarity_score)

        return document_scores


# Example usage
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
query = "this is a query"

similarity_calculator = CalculateSimilarity(documents)
similarity_calculator.preprocess_query(query)
document_scores = similarity_calculator.calculate_similarity()

print("Similarity scores:")
for doc_id, score in enumerate(document_scores):
    print(f"Document {doc_id}: Score {score}")
