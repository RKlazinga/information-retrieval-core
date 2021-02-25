import os

import numpy as np
import pandas as pd
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

    features = np.array([[], [], [], [], [], [], []]).T
    labels = []
    wfdoc, idf, wfcorp = {}, {}, {}

    for idx, (_, query, _, _, _, pos, _, _, _, neg) in tqdm(triples.iterrows()):
        try:
            query = [token.text for token in stem(query)]

            for i, doc in enumerate([neg] + [pos]):
                doc = [token.text for token in stem(doc)]
                intersection = set(query) & set(doc)
                docfeats = np.zeros(7)

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

                features = np.concatenate([features, docfeats[None, :]], axis=0)

                if i == 1:  # we've added a positively labeled doc
                    labels.append(features.shape[0] - 1)
        except:
            print("\n\n\nERRROR")
            print("query", query)
            print("pos", pos)
            print("neg", neg)

    return features, np.array(labels)


def dcg(y_true, y_pred, rank=None):
    order = np.argsort(-y_pred)
    gain = np.take(y_true, order[:rank])
    discounts = np.log2(np.arange(len(gain)) + 2)
    return np.sum(gain / discounts)


def ndcg(y_pred, correct_idxs, rank=None):
    n_q, n_d = len(correct_idxs), len(y_pred)
    y_true = np.zeros((n_q, n_d))
    y_true[np.arange(n_q), correct_idxs] = 1  # for each query, a single 1 where its positive doc is
    score = np.array([dcg(y_true[i], y_pred, rank=rank) for i in range(n_q)])
    best_score = np.array([dcg(np.sort(y_true[i]), np.arange(0, n_d), rank=rank) for i in range(n_q)])
    best_score[best_score == 0] = 1
    ret = score / best_score
    print(ret.mean())
    return ret


if __name__ == "__main__":
    metric = ndcg
    if not os.path.exists("data/features.npy"):
        doc_feats, correct_docs = get_features()
        np.save("data/features.npy", doc_feats)
        np.save("data/labels.npy", correct_docs)
    else:
        doc_feats = np.load("data/features.npy")
        correct_docs = np.load("data/labels.npy")

    y = correct_docs

    print(doc_feats.shape, correct_docs.shape)

    P = np.ones(len(correct_docs)) / len(correct_docs)
    weak_rankers = []binary
    alpha = np.zeros(doc_feats.shape[1])

    weak_ranker_score = []
    for k in range(doc_feats.shape[1]):
        weak_ranker_score.append(ndcg(doc_feats[:, k], correct_docs))

    used_feats = []
    for _ in range(50):
        best_avg = -np.inf
        h = None
        for feat, score in enumerate(weak_ranker_score):
            if feat in used_feats:
                continue
            avg = np.dot(P, score)
            if avg > best_avg:
                h = {"feat": feat, "score": score}
                best_avg = avg
        if h is None:
            break  # all features have been used
        used_feats.append(h["feat"])

        h["alpha"] = 0.5 * (np.log(np.dot(P, 1 + h["score"]) / np.dot(P, 1 - h["score"])))
        weak_rankers.append(h)

        alpha[h["feat"]] += h["alpha"]

        score = ndcg(np.dot(doc_feats, alpha), correct_docs)
        new_P = np.exp(-score)
        P = new_P / new_P.sum()
