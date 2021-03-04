import collections
import numpy as np
import nltk


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
        self.corpussize = len(nltk.word_tokenize(text))

    def _generateNgrams(self, text, n=2):
        """
        Compute an ngram, given the test and N.        
        :param text: The corpus
        :param n: The number, default bigram.
        :return: 
        """
        token = nltk.word_tokenize(text)
        computedNgrams = nltk.util.ngrams(token, n)
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


top100 = {}
with open("data/msmarco-doctest2019-top100") as f:
    for qid, _, docid, _, _, _ in csv.reader(f, delimiter=" "):
        if qid not in top100:
            top100[qid] = []
        top100[qid].append(docid)
preprocess = whoosh.analysis.StemmingAnalyzer()

# with open("data/msmarco-docs.tsv", "rt", encoding="utf8") as FILE:
