import csv
import os
import random

import numpy as np
import pandas as pd
import sklearn.metrics
import whoosh.analysis
import whoosh.scoring
from tqdm import tqdm
from whoosh.filedb.filestore import FileStorage
from util import features_per_doc, getbody

training_set_size = 2500


# The query string for each topicid is querystring[topicid]
querystring = {}
with open("data/msmarco-doctrain-queries.tsv", "rt", encoding="utf8") as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [topicid, querystring_of_topicid] in tsvreader:
        querystring[topicid] = querystring_of_topicid


# For each topicid, the list of positive docids is qrel[topicid]
qrel = {}
with open("data/msmarco-doctrain-qrels.tsv", "rt", encoding="utf8") as f:
    tsvreader = csv.reader(f, delimiter=" ")
    for [topicid, _, docid, rel] in tsvreader:
        assert rel == "1"
        if topicid in qrel:
            qrel[topicid].append(docid)
        else:
            qrel[topicid] = [docid]


def get_features(num_features=None):
    stem = whoosh.analysis.StemmingAnalyzer()

    positive_topics = random.choices(list(querystring.keys()), k=training_set_size)
    positive_samples = {topic: (querystring[topic], qrel[topic]) for topic in positive_topics}

    ix = FileStorage("/HDDs/msmarco").open_index().reader()

    features = np.array([[], [], [], [], [], [], []]).T
    labels = []

    num = 0
    with open("data/msmarco-docs.tsv", "rt", encoding="utf8") as docsfile:
        for _, (query, positives) in tqdm(positive_samples.items()):
            if num_features is not None and num > num_features:
                break

            query = [token.text for token in stem(query)]

            labels.append([])
            for i, docid in enumerate(positives):
                body = getbody(docid, docsfile)
                if body is None:
                    continue
                body = [token.text for token in stem(body)]

                docfeats = features_per_doc(query, body, ix)

                features = np.concatenate([features, docfeats[None, :]], axis=0)
                labels[-1].append(len(features) - 1)

            num += 1

    return features, labels


def dcg(y_true, pi_pred, rank):
    order = np.argsort(-pi_pred)
    gain = y_true[order[:rank]]
    discounts = np.log2(np.arange(len(gain)) + 2)
    return np.sum(gain / discounts)


def ndcg(pi_pred, correct_idxs, rank=None):
    n_q, n_d = len(correct_idxs), len(pi_pred)
    y_true = np.zeros((n_q, n_d))
    for i, query_idxs in enumerate(correct_idxs):
        y_true[i, query_idxs] = 1  # for each query, 1s where its positive docs are

    score = np.array([dcg(y_true[i], pi_pred, rank=rank) for i in range(n_q)])

    ordering = np.arange(0, n_d)
    best_score = np.array(
        [
            dcg(
                np.concatenate((np.zeros(n_d - len(correct_idxs[i])), np.ones(len(correct_idxs[i])))),
                ordering,
                rank=rank,
            )
            for i in range(n_q)
        ]
    )
    best_score[best_score == 0] = 1

    ret = score / best_score
    return ret


def precision(pi_pred, correct_idxs, rank=None):
    n_q, n_d = len(correct_idxs), len(pi_pred)
    y_true = np.zeros((n_q, n_d))
    y_pred = np.zeros_like(y_true)

    scores = []
    order = np.argsort(-pi_pred)
    for i, query_idxs in enumerate(correct_idxs):
        y_true[i, query_idxs] = 1  # for each query, 1s where its positive docs are
        y_pred = y_true[i, order]
        target = np.concatenate((np.zeros(n_d - len(correct_idxs[i])), np.ones(len(correct_idxs[i]))))
        scores.append(sklearn.metrics.precision_score(y_true=target, y_pred=y_pred[:rank]))

    return np.array(scores)


if __name__ == "__main__":
    metric = ndcg

    doc_feats, correct_docs = get_features(num_features=None)

    P = np.ones(len(correct_docs)) / len(correct_docs)
    weak_rankers = []
    alpha = np.zeros(doc_feats.shape[1])

    weak_ranker_score = []
    for k in range(doc_feats.shape[1]):
        weak_ranker_score.append(metric(doc_feats[:, k], correct_docs))

    best_alpha = alpha
    best_score = 0

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

        predictions = np.dot(doc_feats, alpha)
        score = metric(predictions, correct_docs)
        new_P = np.exp(-score)
        P = new_P / new_P.sum()

        if score.mean() > best_score:
            best_alpha = alpha.copy()
            best_score = score.mean()
        print(score.mean())
        print(alpha)
        print(P.min(), P.mean(), P.max())
        print()

    print(best_alpha)
    print(best_score)
    np.save("ada.npy", best_alpha)
