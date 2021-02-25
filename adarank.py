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

    for idx, (_, query, _, _, _, pos, _, _, _, neg) in tqdm(triples.iterrows()):
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

                if i == 1:  # we've added a positively labeled doc
                    labels.append(features.shape[0] - 1)
        except:
            print("\n\n\nERRROR")
            print("query", query)
            print("pos", pos)
            print("neg", neg)

    return features, torch.tensor(labels)


def dcg(relevances):
    relevances = np.asarray(relevances)
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.0
    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)


def ndcg(pi, y, rank=5):
    best_ordering = np.zeros(rank)
    best_ordering[0] = 1
    best_dcg = dcg(best_ordering)

    y_onehot = np.zeros(len(pi))
    y_onehot[y] = 1

    if best_dcg == 0:
        return 0.0

    return dcg(y_onehot[pi[:rank]]) / best_dcg


class WeakRanker(torch.nn.Module):
    def __init__(self, weighting, features, labels):
        super().__init__()

        self.feature_idx = 0

        max_val = -1e30
        for k in range(features.shape[1]):
            permutation = self.sort(features, k)
            val = 0
            for i, l in enumerate(labels):
                val += weighting[i] * metric(permutation, l)
            print(val)
            if max_val < val.item():
                max_val = val
                self.feature_idx = k
        print("Best feature:", self.feature_idx)

    def forward(self, x, k=None):
        # return scores of x
        if k is not None:
            return x[:, k]
        else:
            return x[:, self.feature_idx]

    def sort(self, x, k=None):
        return torch.argsort(self.forward(x, k))


class BoostedRanker(torch.nn.Module):
    def __init__(self, alphas, hs):
        super().__init__()
        self.alphas = alphas
        self.hs = hs

    def forward(self, x):
        # return ranking of x
        out = torch.zeros((x.shape[0]))
        for weight, learner in zip(self.alphas, self.hs):
            out += weight * learner(x)
        return out

    def sort(self, x):
        return torch.argsort(self.forward(x))


if __name__ == "__main__":
    metric = ndcg
    if not os.path.exists("data/features.pth"):
        doc_feats, correct_docs = get_features()

        torch.save(doc_feats, "data/features.pth")
        torch.save(correct_docs, "data/labels.pth")
    else:
        doc_feats = torch.load("data/features.pth")
        correct_docs = torch.load("data/labels.pth")

    print(doc_feats.shape, correct_docs.shape)

    Ps = [torch.ones(len(correct_docs)) / len(correct_docs)]

    alphas = []
    hs = []

    depth = 20

    for t in range(depth):
        h = WeakRanker(Ps[-1], doc_feats, correct_docs)
        hs.append(h)

        ordering_idxs = h.sort(doc_feats)

        alpha = 0
        for i, correct_idx in enumerate(correct_docs):
            alpha += (
                Ps[-1][i]
                * (1 + metric(ordering_idxs, correct_idx))
                / Ps[-1][i]
                * (1 - metric(ordering_idxs, correct_idx))
            )
        alphas.append(torch.log(alpha) / 2)

        f = BoostedRanker(alphas, hs)

        ordering_idxs = f.sort(doc_feats)

        Ps.append(torch.zeros(len(correct_docs)))
        norm = 0
        for i, correct_idx in enumerate(correct_docs):
            print(ordering_idxs[:5], correct_idx)
            weight = np.exp(-metric(ordering_idxs, correct_idx))
            Ps[-1][i] = weight
            norm += weight
        Ps[-1] /= norm

