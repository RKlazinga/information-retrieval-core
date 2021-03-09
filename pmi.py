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
    def __init__(self, tokens):
        self.corpussize = len(tokens)
        print(self.corpussize)
        self.bigrams = collections.Counter(ngrams(tokens, 2))
        self.unigrams = collections.Counter(ngrams(tokens, 1))

    def _count_for_bigram(self, word1, word2):
        return self.bigrams[(word1, word2)]

    def _count_for_unigram(self, word1):
        count = self.unigrams[(word1)]
        if count == 0:
            count = 0.001
        return count

    def compute(self, word1, word2):
        P_w1w2 = self._count_for_bigram(word1, word2) / self.corpussize
        p_w1 = self._count_for_unigram(word1) / self.corpussize
        p_w2 = self._count_for_unigram(word2) / self.corpussize
        score = np.log(P_w1w2 / (p_w1 * p_w2))
        if np.isfinite(score):
            return score
        else:
            return -20


if __name__ == "__main__":
    docids = set()
    with open("data/msmarco-doctrain-top100") as f:
        for _, _, docid, _, _, _ in csv.reader(f, delimiter=" "):
            docids.add(docid)

    corpus = []
    with open("data/msmarco-docs.tsv", "rt", encoding="utf8") as FILE:
        for docid in random.choices(list(docids), k=1_000_000):
            body = getbody(docid, FILE)
            if body is None:
                continue
            corpus.append(body)

    pmi = PMI([token.text for token in preprocess(" ".join(corpus))])
    joblib.dump(pmi, "pmi.pkl")
