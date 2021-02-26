import csv
import random

import numpy as np
import pandas as pd
import whoosh.analysis
import whoosh.scoring
from tqdm import tqdm
from whoosh.filedb.filestore import FileStorage


training_set_size = 1000


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

    # triples = pd.read_csv(
    #     "data/triples.tsv",
    #     sep="\t",
    #     names=["topic", "query", "pos1", "pos2", "pos3", "pos4", "not1", "not2", "not3", "not4"],
    # )
    # print(triples)

    ix = FileStorage("data/quickidx").open_index().reader()
    ndocs = ix.doc_count_all()
    avg_doc_len = ix.field_length("body") / ndocs

    features = np.array([[], [], [], [], [], [], []]).T
    labels = []
    wfdoc, idf, wfcorp = {}, {}, {}

    num = 0
    for idx, (query, positives) in tqdm(positive_samples.items()):
        if num_features is not None and num > num_features:
            break
        try:
            query = [token.text for token in stem(query)]

            labels.append([])
            for i, doc in enumerate(positives):
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
                num += 1

                labels[-1].append(i)
        except:
            print("\n\n\nERRROR")
            print("query", query)
            print("pos", positives)

    return features, np.array(labels)


def dcg(y_true, y_pred, rank):
    # print(y_pred)
    order = np.argsort(-y_pred)
    gain = np.take(y_true, order[:rank])
    # print(gain)
    discounts = np.log2(np.arange(len(gain)) + 2)
    # print(gain / discounts)
    # print(np.sum(gain / discounts), "\n\n\n")
    return np.sum(gain / discounts)


def ndcg(y_pred, correct_idxs, rank=None):
    n_q, n_d = len(correct_idxs), len(y_pred)
    y_true = np.zeros((n_q, n_d))
    y_true[correct_idxs] = 1  # for each query, a single 1 where its positive doc is

    # print("score")
    score = np.array([dcg(y_true[i], y_pred, rank=rank) for i in range(n_q)])

    # print("best")
    best_score = np.array([dcg(np.sort(y_true[i]), np.arange(0, n_d), rank=rank) for i in range(n_q)])
    best_score[best_score == 0] = 1

    ret = score / best_score
    # print(ret.mean())
    return ret


if __name__ == "__main__":
    metric = ndcg
    # if not os.path.exists("data/features.npy"):
    doc_feats, correct_docs = get_features(num_features=None)
    np.save("data/features.npy", doc_feats)
    np.save("data/labels.npy", correct_docs)
    # else:
    #     doc_feats = np.load("data/features.npy")
    #     correct_docs = np.load("data/labels.npy")

    y = correct_docs

    print(doc_feats.shape, correct_docs.shape)

    P = np.ones(len(correct_docs)) / len(correct_docs)
    weak_rankers = []
    alpha = np.zeros(doc_feats.shape[1])

    weak_ranker_score = []
    for k in range(doc_feats.shape[1]):
        weak_ranker_score.append(ndcg(doc_feats[:, k], correct_docs))

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

        score = ndcg(np.dot(doc_feats, alpha), correct_docs)
        new_P = np.exp(-score)
        P = new_P / new_P.sum()

        if score.mean() > best_score:
            best_alpha = alpha.copy()
            best_score = score.mean()
        print(score.mean(), alpha)

    print(best_alpha)
    print(best_score)
