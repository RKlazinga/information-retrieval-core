import os

import numpy as np
import pandas as pd
import torch
import whoosh.analysis
import whoosh.scoring
from tqdm import tqdm
from whoosh import index, writing
from whoosh.filedb.filestore import FileStorage


def get_features():
    stem = whoosh.analysis.StemmingAnalyzer()

    triples = pd.read_csv(
        "data/triples.tsv",
        sep="\t",
        names=["topic", "query", "pos1", "pos2", "pos3", "pos4", "not1", "not2", "not3", "not4"],
    )
    print(triples)

    ix = FileStorage("data/quickidx").open_index().reader()
    ndocs = ix.doc_count_all()
    avg_doc_len = ix.field_length("body") / ndocs

    features = torch.tensor([[], [], [], [], [], [], []]).T
    labels = []
    wfdoc, idf, wfcorp = {}, {}, {}

    for _, (_, query, _, _, _, pos, _, _, _, neg) in tqdm(triples.iterrows()):
        try:
            query = [token.text for token in stem(query)]

            for i, doc in enumerate([neg] + [pos]):
                doc = [token.text for token in stem(doc)]
                intersection = set(query) & set(doc)
                docfeats = torch.zeros(7)

                for term in intersection:
                    if term not in wfdoc:
                        wfdoc[term] = doc.count(term)
                        idf[term] = np.log(ndocs / (ix.doc_frequency("body", term) + 1e-8))
                        wfcorp[term] = ix.frequency("body", term) + 1e-8

                    docfeats[0] += np.log(wfdoc[term] + 1)
                    docfeats[1] += np.log(idf[term])
                    docfeats[2] += np.log(wfdoc[term] / len(doc) * idf[term] + 1)
                    docfeats[3] += np.log(ndocs / wfcorp[term] + 1)
                    docfeats[4] += np.log(wfdoc[term] / len(doc) + 1)
                    docfeats[5] += np.log(wfdoc[term] * ndocs / (len(doc) * wfcorp[term]) + 1)
                    docfeats[6] += whoosh.scoring.bm25(idf[term], wfcorp[term], len(doc), avg_doc_len, B=0.75, K1=1.2)

                if len(intersection) > 0:
                    docfeats[6] = np.log(docfeats[6])

                features = torch.cat([features, docfeats[None, :]], axis=0)

                labels.append(i)
        except:
            print("\n\n\nERRROR")
            print("query", query)
            print("pos", pos)
            print("neg", neg)

    return features, torch.tensor(labels)


def dcg(relevances, rank):
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.0
    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)


def ndcg(relevances, rank):
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.0
    return dcg(relevances, rank) / best_dcg


def metric(pi, y):
    return ndcg(relevances=y.numpy()[pi.numpy()], rank=len(pi) - 1)


class WeakRanker(torch.nn.Module):
    def __init__(self, P, x, y):
        super().__init__()

        self.feature_idx = 0

        max_val = -1e30
        for k in range(x.shape[1]):
            if max_val < (val := torch.sum(P * metric(self.sort(x, k), y))):
                max_val = val
                self.feature_idx = k

    def forward(self, x, k=None):
        # return scores of x
        if k is not None:
            return x[:, k]
        else:
            return x[:, self.feature_idx]

    def sort(self, x, k=None):
        pi = torch.argsort(self.forward(x, k))
        return pi


class BoostedRanker(torch.nn.Module):
    def __init__(self, alphas, hs):
        super().__init__()
        self.alphas = alphas
        self.hs = hs

    def forward(self, x):
        # return ranking of x
        out = torch.zeros((x.shape[0]))
        for alpha, h in zip(self.alphas, self.hs):
            out += alpha * h(x)
        return out

    def sort(self, x):
        # evaluate all docs and sort by score
        pi = torch.argsort(self.forward(x))
        return pi


if __name__ == "__main__":
    if not os.path.exists("data/features.pth"):
        X, y = get_features()
        torch.save(X, "data/features.pth")
        torch.save(y, "data/labels.pth")
    else:
        X = torch.load("data/features.pth")
        y = torch.load("data/labels.pth")

    P = torch.ones(len(y)) / len(y)

    alphas = []
    hs = []

    depth = 20

    for t in range(depth):
        h = WeakRanker(P, X, y)
        hs.append(h)

        pi = h.sort(X)

        alpha = 1 / 2 * torch.log((torch.sum(P * (1 + metric(pi, y)))) / (P * (1 - metric(pi, y))))
        alphas.append(alpha)

        f = BoostedRanker(alphas, hs)

        pi = f.sort(X)

        P = torch.nn.functional.softmax(metric(pi, y))  # hier gaat iets mis
