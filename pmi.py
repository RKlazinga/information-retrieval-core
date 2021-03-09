import collections
import csv
import multiprocessing as mp
import random
import time

import numpy as np
import whoosh.analysis
from nltk.util import ngrams
from tqdm import tqdm

import util

pbar = None


def count(x):
    return collections.Counter(x)


def fastcount(grams):
    with mp.Pool(processes=24) as pool:
        counts = collections.Counter()
        for result in tqdm(pool.imap_unordered(func=count, iterable=util.chunks(grams))):
            counts += result
            pbar.update(CHUNK_SIZE)
    return counts


preprocess = whoosh.analysis.StemmingAnalyzer()


class PMI:
    def __init__(self, tokens):
        global pbar
        self.corpussize = len(tokens)
        print(self.corpussize)
        tik = time.time()
        print("counting bigrams...")
        pbar = tqdm(total=self.corpussize)
        self.bigrams = fastcount(ngrams(tokens, 2))
        print("counting unigrams...")
        pbar = tqdm(total=self.corpussize)
        self.unigrams = fastcount(ngrams(tokens, 1))
        print("took:", time.time() - tik)

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
    from features import getbody

    print("loading in documents...")
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

    print("preprocessing corpus...")
    pmi = PMI([token.text for token in preprocess(" ".join(corpus))])
    util.save(pmi, "pmi.pkl")
