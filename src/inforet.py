import xml.etree.ElementTree as ET
import re
import math
from collections import defaultdict, Counter
import numpy as ny
import os
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag
from nltk import download as nltk_download

# Download required NLTK resources
nltk_download('punkt')
nltk_download('averaged_perceptron_tagger')
nltk_download('wordnet')
nltk_download('stopwords')

# Define the output folder and create it if it doesn't already exist
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.getcwd()), "results")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# File paths
DOCUMENTS_FILE = r"C:\Users\afrat\OneDrive\Desktop\mos\Mechanics_of_Search_ASS1\data\cranfield-trec-dataset-main\\cran.all.1400.xml"
QUERIES_FILE = r"C:\Users\afrat\OneDrive\Desktop\mos\Mechanics_of_Search_ASS1\data\cranfield-trec-dataset-main\cran.qry.xml"


"""Handles text preprocessing tasks like tokenization, stemming, and lemmatization."""
class TextPreProcessor:

    def __init__(self, use_stemming=True, use_lemmatization=True):
        self.pStemmer = PorterStemmer() if use_stemming else None
        self.wnLem = WordNetLemmatizer() if use_lemmatization else None
        self.sw = set(stopwords.words('english'))

    def preprocesstext(self, text):
        """
        Preprocesses the input text by:
        1. Converting it to lowercase
        2. Removing punctuation
        3. Tokenizing the text
        4. Removing stopwords
        5. Applying lemmatization or stemming as configured
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) 
        tokens = text.split()
        pTags = pos_tag(tokens)
        processed_tokens = []
        for word, tag in pTags:
            if word not in self.sw:
                if self.wnLem:
                    lemma = self.wnLem.lemmatize(word, self.map_pos_tag_to_wordnet(tag))
                else:
                    lemma = word
                if self.pStemmer:
                    processed_tokens.append(self.pStemmer.stem(lemma))
                else:
                    processed_tokens.append(lemma)
        return processed_tokens

    @staticmethod
    def map_pos_tag_to_wordnet(treebank_tag):
        """
        Maps a treebank POS tag to a WordNet POS tag.
        This is required for proper lemmatization as WordNet needs a specific POS format.
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN


class XmlDocumentParser:
    """Parses XML documents and queries."""

    def __init__(self, doc_tag, text_tag):
        self.doc_tag = doc_tag
        self.text_tag = text_tag

    def parse_doc(self, file_path):
        """
        Parses the XML file at the given file_path, extracts the text content
        from the elements defined by doc_tag and text_tag, and returns a list 
        of the extracted text. If no text is found, it adds an empty string.
        """        
        tree = ET.parse(file_path)
        root = tree.getroot()
        data = []
        for element in root.findall(self.doc_tag):
            text_element = element.find(self.text_tag)
            if text_element is not None and text_element.text is not None:
                data.append(text_element.text.strip())
            else:
                data.append("")
        return data


class InvertedIndexBuilder:
    """Builds and manages the inverted index for document retrieval."""

    def __init__(self):
        self.vocab = set()
        self.doc_freq = Counter()
        self.term_doc_matrix = defaultdict(lambda: defaultdict(int))
        self.term_to_index = {}

    def build(self, processed_docs):
        """
        Builds the inverted index from the list of processed documents. 
        This method updates:
        - vocab: Unique terms from the documents
        - doc_freq: Number of documents a term appears in
        - term_doc_matrix: The frequency of terms in each document
        - term_to_index: Mapping of terms to indices in the vocabulary
        """
        for doc_id, doc in enumerate(processed_docs):
            unique_terms = set(doc)
            self.vocab.update(unique_terms)
            for term in unique_terms:
                self.doc_freq[term] += 1
            for term in doc:
                if term not in self.term_to_index:
                    self.term_to_index[term] = len(self.term_to_index)
                term_id = self.term_to_index[term]
                self.term_doc_matrix[term_id][doc_id] += 1

    def get_vocab(self):
        """Returns the vocabulary as a list."""
        return list(self.vocab)

    def get_doc_freq(self):
        """Returns the document frequency dictionary."""
        return self.doc_freq

    def get_term_doc_matrix(self):
        """Returns the term-document matrix."""
        return self.term_doc_matrix

    def get_term_to_index(self):
        """Returns the term-to-index mapping."""
        return self.term_to_index


class InfoRetrievalModels:
    """Implements retrieval models like VSM, BM25, and Unigram Language Model."""

    def __init__(self, index, num_docs, processed_docs):
        self.index = index
        self.num_docs = num_docs
        self.processed_docs = processed_docs
        self.avg_doc_length = ny.mean([len(doc) for doc in processed_docs])
        self.doc_lengths = [len(doc) for doc in processed_docs]
        self.vocab_size = len(self.index.get_vocab())

    def calc_tf_idf(self, query):
        """
        Computes the TF-IDF (Term Frequency-Inverse Document Frequency) vector for a query.
        - query: A list of terms in the query.
        """
        query_vector = ny.zeros(len(self.index.get_vocab()))
        term_to_index = self.index.get_term_to_index()
        query_length = len(query)
        if query_length == 0:
            return query_vector
        for term in query:
            if term in term_to_index:
                tf = query.count(term) / query_length
                df = self.index.get_doc_freq().get(term, 1)
                idf = math.log(self.num_docs / (df + 1))
                query_vector[term_to_index[term]] = tf * idf
        query_norm = ny.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector /= query_norm
        return query_vector

    def calc_cosine_similarity(self, query_vector, term_doc_matrix, doc_lengths, num_docs):
        """
        Computes cosine similarity scores between a query and all documents.
        - query_vector: The TF-IDF vector of the query.
        - term_doc_matrix: The term-document matrix (term frequency in each document).
        - doc_lengths: The lengths of the documents.
        - num_docs: Total number of documents.
        """
        scores = ny.zeros(num_docs)
        query_norm = ny.linalg.norm(query_vector)
        if query_norm == 0:
            return scores
        for term_id, weight in enumerate(query_vector):
            if weight == 0:
                continue
            for doc_id, tf in term_doc_matrix[term_id].items():
                scores[doc_id] += weight * tf
        for doc_id in range(num_docs):
            if doc_lengths[doc_id] == 0:
                scores[doc_id] = 0
            else:
                scores[doc_id] /= (query_norm * doc_lengths[doc_id])
        return scores

    def calc_bm25(self, query, k1=2.5, b=0.8):
        """
        Computes BM25 scores for a query.
        - query: The query containing terms to match.
        - k1: The term frequency scaling factor (default 2.5).
        - b: The document length scaling factor (default 0.8).
        """
        scores = ny.zeros(self.num_docs)
        term_to_index = self.index.get_term_to_index()
        for term in query:
            if term in term_to_index:
                term_id = term_to_index[term]
                df = self.index.get_doc_freq().get(term, 1)
                idf = math.log((self.num_docs - df + 0.5) / (df + 0.5))
                for doc_id, tf in self.index.get_term_doc_matrix()[term_id].items():
                    doc_length = len(self.processed_docs[doc_id])
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
                    scores[doc_id] += idf * (numerator / denominator)
        return scores

    def gen_unigram_language_model(self, query, alpha=0.1):
        """
        Computes unigram language model scores for a query using different smoothing methods.
        - query: The query containing terms to match.
        - alpha: The smoothing parameter (default 0.1).
        """
        scores = ny.zeros(self.num_docs)
        term_to_index = self.index.get_term_to_index()
        vocab_size = self.vocab_size

        for doc_id in range(self.num_docs):
            doc_length = self.doc_lengths[doc_id]
            doc_score = 0.0
            for term in query:
                if term in term_to_index:
                    term_id = term_to_index[term]
                    tf = self.index.get_term_doc_matrix()[term_id].get(doc_id, 0)
                    #laplace smoothing
                    doc_score += math.log((tf + 1) / (doc_length + vocab_size))
                else:
                    #laplace smoothing
                    doc_score += math.log(1 / (doc_length + vocab_size))
            scores[doc_id] = doc_score
        return scores


class ResultsWriter:
    """Handles writing results to output files."""

    @staticmethod
    def generate_trec_output(ranked_docs, scores, query_id, run_id):
        """
        Generates TREC evaluation output lines.
        - ranked_docs: A list of document IDs ranked by relevance.
        - scores: A list of scores corresponding to each document ID.
        - query_id: The ID of the query for which results are generated.
        - run_id: The ID representing the run (e.g., a specific retrieval model).
        """
        output_lines = []
        for rank, doc_id in enumerate(ranked_docs, start=1):
            score = scores[doc_id]
            if ny.isnan(score):
                score = 0.0
            output_lines.append(f"{query_id} Q0 {doc_id + 1} {rank} {score:.6f} {run_id}")
        return output_lines

    @staticmethod
    def write_output(output_lines, output_file):
        """Writes output lines to a file."""
        with open(output_file, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"Output file '{output_file}' generated successfully.")


class IRSystem:
    """Orchestrates the entire information retrieval process."""

    def __init__(self):
        self.text_processor = TextPreProcessor()
        self.document_parser = XmlDocumentParser('doc', 'text')
        self.query_parser = XmlDocumentParser('top', 'title')
        self.index = InvertedIndexBuilder()
        self.results_writer = ResultsWriter()

    def run(self):
        """Runs the information retrieval system."""
        
        """ Parse documents and queries """
        documents = self.document_parser.parse_doc(DOCUMENTS_FILE)
        queries = self.query_parser.parse_doc(QUERIES_FILE)

        # Preprocess documents and queries
        processed_docs = [self.text_processor.preprocesstext(doc) for doc in documents]
        processed_queries = [self.text_processor.preprocesstext(query) for query in queries]

        # Build the inverted index
        self.index.build(processed_docs)
        num_docs = len(processed_docs)
        num_terms = len(self.index.get_vocab())

        # Precompute document lengths for cosine similarity
        doc_lengths = ny.zeros(num_docs)
        for term_id in range(num_terms):
            for doc_id, tf in self.index.get_term_doc_matrix()[term_id].items():
                doc_lengths[doc_id] += tf * tf
        doc_lengths = ny.sqrt(doc_lengths)

        # Initialize retrieval models
        retrieval_models = InfoRetrievalModels(self.index, num_docs, processed_docs)

        # Process queries and generate output for VSM with expansion
        vsm_output = []
        for query_id, query in enumerate(processed_queries, start=1):
            query_vector = retrieval_models.calc_tf_idf(query)
            scores = retrieval_models.calc_cosine_similarity(query_vector, self.index.get_term_doc_matrix(), doc_lengths, num_docs)
            ranked_docs = ny.argsort(scores)[::-1]
            expanded_query = self.expand_query_for_perf(query, ranked_docs, processed_docs, k=5)
            query_vector = retrieval_models.calc_tf_idf(expanded_query)
            scores = retrieval_models.calc_cosine_similarity(query_vector, self.index.get_term_doc_matrix(), doc_lengths, num_docs)
            ranked_docs = ny.argsort(scores)[::-1]
            vsm_output.extend(self.results_writer.generate_trec_output(ranked_docs, scores, query_id, 'VSM'))
        self.results_writer.write_output(vsm_output, os.path.join(OUTPUT_FOLDER, "vsm_results.txt"))

        # Process queries and generate output for BM25 with expansion
        bm25_output = []
        for query_id, query in enumerate(processed_queries, start=1):
            scores = retrieval_models.calc_bm25(query)
            ranked_docs = ny.argsort(scores)[::-1]
            expanded_query = self.expand_query_for_perf(query, ranked_docs, processed_docs, k=5)
            scores = retrieval_models.calc_bm25(expanded_query)
            ranked_docs = ny.argsort(scores)[::-1]
            bm25_output.extend(self.results_writer.generate_trec_output(ranked_docs, scores, query_id, 'BM25'))
        self.results_writer.write_output(bm25_output, os.path.join(OUTPUT_FOLDER, "bm25_results.txt"))

        # Process queries and generate output for Unigram Language Model
        unigram_output = []
        for query_id, query in enumerate(processed_queries, start=1):
            scores = retrieval_models.gen_unigram_language_model(query)
            ranked_docs = ny.argsort(scores)[::-1]
            unigram_output.extend(self.results_writer.generate_trec_output(ranked_docs, scores, query_id, 'UNIGRAM'))
        self.results_writer.write_output(unigram_output, os.path.join(OUTPUT_FOLDER, "unigram_results.txt"))

    @staticmethod
    def expand_query_for_perf(query, top_docs, processed_docs, k=5):
        """Expands the query with terms from top-k documents."""
        expanded_query = query.copy()
        for doc_id in top_docs[:k]:
            expanded_query.extend(processed_docs[doc_id])
        return expanded_query


if __name__ == "__main__":
    ir_system = IRSystem()
    ir_system.run()