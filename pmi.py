import collections
import csv
import random

import joblib
import numpy as np
import whoosh.analysis
from nltk.util import ngrams
from tqdm import tqdm

from features import getbody

preprocess = whoosh.analysis.StemmingAnalyzer()


class PMI:
    def __init__(self, text):
        self.tokens = list(preprocess(text))
        self.corpussize = len(self.tokens)
        print(self.corpussize)
        self._generate_bigrams(text)
        self._generate_unigrams(text)

    def _generate_ngrams(self, text, n=2):
        computedNgrams = ngrams(self.tokens, n)
        return collections.Counter(computedNgrams)

    def _generate_bigrams(self, text):
        self.bigrams = self._generate_ngrams(text, 2)

    def _generate_unigrams(self, text):
        self.unigrams = self._generate_ngrams(text, 1)

    def _count_for_bigram(self, word1, word2):
        return self.bigrams[(word1, word2)]

    def _count_for_unigram(self, word1):
        count = self.unigrams[(word1)]
        if count == 0:
            count = 0.001
        return count

    def compute(self, word1, word2):
        pmi = 0
        if word1 is not None and word2 is not None:
            P_w1w2 = self._count_for_bigram(word1, word2) / self.corpussize
            p_w1 = self._count_for_unigram(word1) / self.corpussize
            p_w2 = self._count_for_unigram(word2) / self.corpussize
            try:
                pmi = np.log2(P_w1w2 / (p_w1 * p_w2))
            except ValueError:
                pmi = 99999
        return pmi


docids = set()
with open("data/msmarco-doctrain-top100") as f:
    for _, _, docid, _, _, _ in csv.reader(f, delimiter=" "):
        docids.add(docid)

corpus = []
with open("data/msmarco-docs.tsv", "rt", encoding="utf8") as FILE:
    for docid in random.choices(list(docids), k=100_000):
        body = getbody(docid, FILE)
        if body is None:
            continue
        corpus.append(body)

pmi = PMI(" ".join(corpus))
del pmi.tokens
joblib.dump(pmi, "pmi.pkl")
