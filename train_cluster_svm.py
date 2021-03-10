import argparse
import csv
import multiprocessing as mp
import os
import random

import hdbscan
import numpy as np
import sklearn.cluster
import sklearn.metrics as metrics
import sklearn.svm
import whoosh.analysis
import whoosh.scoring
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from whoosh.filedb.filestore import FileStorage

import util
from features import get_doc_feats, get_query_features
from query_cluster import get_query_features
from singletons import PREPROCESS, QREL, QUERYDICT, TOP100


def get_features(inp):
    features = []
    labels = []
    queryfeats = []

    for _ in inp:
        try:
            qid = random.choice(list(QUERYDICT.keys()))
            query = [token.text for token in PREPROCESS(QUERYDICT[qid])]
            qf = get_query_features(query)

            # positive sample
            docid = random.choice(QREL[qid])
            docfeats = get_doc_feats(docid, query, FILE)
            features.append(docfeats)
            if not args.graded:
                labels.append(1)
            else:
                labels.append(4 if random.random() < 0.5 else 3)
            queryfeats.append(qf)

            # negative sample
            docid = random.choice(list(set(TOP100[qid]) - set(QREL[qid])))
            docfeats = get_doc_feats(docid, query, FILE)
            features.append(docfeats)
            if not args.graded:
                labels.append(0)
            else:
                labels.append(1 if random.random() > 0.5 else 2)
            queryfeats.append(qf)

        except Exception as e:
            print("ERROR:", e)

    return np.array(queryfeats), np.array(features), np.array(labels)


class ClusterSVM:
    def __init__(self, qf, df, labels, verbose=True):
        self.cluster = hdbscan.HDBSCAN(min_cluster_size=10, prediction_data=True).fit(qf)
        clusids, strengths = hdbscan.approximate_predict(self.cluster, qf)
        uniques = np.sort(np.unique(clusids))
        n_labels = len(uniques)
        if verbose:
            label_strengths = [np.median(strengths[clusids == l]) for l in uniques]
            label_counts = [np.sum(clusids == l) for l in uniques]
            print("# clusters found:", n_labels)
            print(
                f"cluster sizes:   min:{np.min(label_counts)}   mean:{np.mean(label_counts)}   max:{np.max(label_counts)}"
            )
            print(
                f"median label strengths:   min:{np.min(label_strengths)}   mean:{np.mean(label_strengths)}   max:{np.max(label_strengths)}"
            )

        self.svms = [sklearn.svm.SVR() for _ in range(n_labels)]
        for l, svm in zip(uniques, self.svms):
            indices = clusids == l
            print(l, indices.sum())
            svm.fit(df[indices], labels[indices])

    def predict(self, qf, df):
        clusids, _ = hdbscan.approximate_predict(self.cluster, qf)
        preds = []
        for i, l in enumerate(clusids):
            preds.append(self.svms[l].predict(df[[i]]))
        return np.concatenate(preds)


def process_graded(inp):
    qid, _, docid, relevance = inp
    query = [token.text for token in PREPROCESS(QUERYDICT[qid])]
    queryfeats = get_query_features(query)
    docfeats = get_doc_feats(docid, query, FILE)
    return queryfeats, docfeats, relevance


def opendocsfile():
    global FILE
    FILE = open("data/msmarco-docs.tsv", "rt", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-graded", action="store_true")
    parser.add_argument("-num_docs", type=int, default=25_000)
    parser.add_argument("-num_graded", type=int, default=25_000)
    parser.add_argument("-noload", action="store_true")
    args = parser.parse_args()

    feature_file = f"querydoc_features_{args.num_docs + (args.num_graded if args.graded else 0)}_{'graded' if args.graded else 'binary'}.pkl"
    if args.noload or not os.path.exists(feature_file):
        print("getting features...")
        queryfeats, features, labels = [], [], []
        pbar = tqdm(total=args.num_docs)
        chunksize = 10
        with mp.Pool(24, initializer=opendocsfile) as pool:
            for q, f, l in pool.imap_unordered(get_features, util.chunks(range(int(args.num_docs / 2)), n=chunksize)):
                queryfeats.append(q.squeeze())
                features.append(f.squeeze())
                labels.append(l.squeeze())
                pbar.update(chunksize * 2)
            queryfeats = np.concatenate(queryfeats)
            features = np.concatenate(features)
            labels = np.concatenate(labels)

            if args.graded:
                with open("graded-qrels.tsv", "rt", encoding="utf8") as gradqrelsfile:
                    pbar = tqdm(total=316)
                    graded_queryfeats, graded_features, graded_labels = [], [], []
                    with mp.Pool(24) as pool:
                        for queryfeats, docfeats, relevance in pool.imap_unordered(
                            process_graded, csv.reader(gradqrelsfile, delimiter=" ")
                        ):
                            graded_queryfeats.append(queryfeats)
                            graded_features.append(docfeats)
                            graded_labels.append(relevance)
                            pbar.update()
                for _ in range(args.num_graded // 316):
                    queryfeats = np.concatenate((features, graded_queryfeats), axis=0)
                    features = np.concatenate((features, graded_features), axis=0)
                    labels = np.concatenate((labels, graded_labels), axis=0)
                labels = labels.astype(np.float32)
        util.save(feature_file, (queryfeats, features, labels))
    else:
        queryfeats, features, labels = util.load(feature_file)

    print("Q", queryfeats)
    print("X", features)
    print("Y", labels)
    print(queryfeats.shape, features.shape, labels.shape)

    trainQ, testQ, trainX, testX, trainY, testY = train_test_split(queryfeats, features, labels, test_size=0.1)

    print("training...")
    clusvm = ClusterSVM(trainQ, trainX, trainY)
    trainPred = clusvm.predict(trainQ, trainX)
    testPred = clusvm.predict(testQ, testX)

    print(testPred)

    print("Train MSE:", metrics.mean_squared_error(trainPred, trainY))
    print("Train score (exact matches):", np.mean(trainPred == trainY))

    print("Test MSE:", metrics.mean_squared_error(testPred, testY))
    print("Test R^2:", metrics.r2_score(testPred, testY))
    print("Test MAE:", metrics.median_absolute_error(testPred, testY))
    print("Test score (exact matches):", np.mean(testPred == testY))

    util.save("clusvm.pkl", clusvm)
