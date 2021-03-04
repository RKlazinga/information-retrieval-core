import collections
import csv

import joblib
from nltk.util import ngrams
import numpy as np
import whoosh.analysis

from features import getbody

preprocess = whoosh.analysis.StemmingAnalyzer()


class PMI:
    bigrams = None
    unigrams = None
    corpussize = None

    def __init__(self, text):
        """
        Features for PMI.
        Will init corpus.
        :param text: The corpus
        """
        self._generateBigrams(text)
        self._generateUnigrams(text)
        self.corpussize = len(preprocess(text))

    def _generateNgrams(self, text, n=2):
        """
        Compute an ngram, given the test and N.        
        :param text: The corpus
        :param n: The number, default bigram.
        :return: 
        """
        token = preprocess(text)
        computedNgrams = ngrams(token, n)
        return collections.Counter(computedNgrams)

    def _generateBigrams(self, text):
        """
        Generate and store the bigrams        
        :param text: corpus
        :return: 
        """
        self.bigrams = self._generateNgrams(text, 2)

    def _generateUnigrams(self, text):
        """
        Generate and store the unigrams        
        :param text: corpus
        :return: 
        """
        self.unigrams = self._generateNgrams(text, 1)

    def _getCountForBigram(self, word1, word2):
        """
        Return the count of occurances for bigram        
        :param word1: 
        :param word2: 
        :return: 
        """
        return self.bigrams[(word1, word2)]

    def _getCountForUnigram(self, word1):
        """
        Return the count of occurances for bigram        
        :param word1: 
        :return: 
        """
        count = self.unigrams[(word1)]
        if count == 0:
            count = 0.001
        return count

    def compute(self, word1, word2):
        """
        Compute the PMI value of a bigram        
        :param word1: 
        :param word2: 
        :return: 
        """
        pmi = 0
        if word1 is not None and word2 is not None:
            P_w1w2 = self._getCountForBigram(word1, word2) / self.corpussize
            p_w1 = self._getCountForUnigram(word1) / self.corpussize
            p_w2 = self._getCountForUnigram(word2) / self.corpussize
            try:
                pmi = np.log2(P_w1w2 / (p_w1 * p_w2))
            except ValueError:
                pmi = 99999
        return pmi


top100 = set()
with open("data/msmarco-doctest2019-top100") as f:
    for qid, _, docid, _, _, _ in csv.reader(f, delimiter=" "):
        top100.add(docid)

text = ""
with open("data/msmarco-docs.tsv", "rt", encoding="utf8") as FILE:
    for docid in top100:
        body = getbody(docid, FILE)
        if body is None:
            continue
        text += " " + body

pmi = PMI(text)
joblib.dump(pmi, "pmi.pkl")
