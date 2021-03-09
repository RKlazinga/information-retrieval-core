import argparse
import csv
import multiprocessing as mp
import os
import random

import joblib
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import whoosh.analysis
import whoosh.scoring
from sklearn import svm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from whoosh.filedb.filestore import FileStorage

from features import get_doc_feats
from pmi import PMI

if not os.path.exists("querystrings.pkl"):
    # if True:
    # The query string for each topicid is querystring[topicid]
    querystring = {}
    with open("data/msmarco-doctrain-queries.tsv", "rt", encoding="utf8") as f:
        for [topicid, querystring_of_topicid] in csv.reader(f, delimiter="\t"):
            querystring[topicid] = querystring_of_topicid
    with open("data/msmarco-docdev-queries.tsv", "rt", encoding="utf8") as f:
        for [topicid, querystring_of_topicid] in csv.reader(f, delimiter="\t"):
            querystring[topicid] = querystring_of_topicid

    # For each topicid, the list of positive docids is qrel[topicid]
    qrel = {}
    with open("data/msmarco-doctrain-qrels.tsv", "rt", encoding="utf8") as f:
        for [topicid, _, docid, rel] in csv.reader(f, delimiter=" "):
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
    with open("data/msmarco-docdev-qrels.tsv", "rt", encoding="utf8") as f:
        for [topicid, _, docid, rel] in csv.reader(f, delimiter=" "):
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]

    top100 = {}
    with open("data/msmarco-doctrain-top100") as f:
        for qid, _, docid, _, _, _ in csv.reader(f, delimiter=" "):
            if qid not in top100:
                top100[qid] = []
            top100[qid].append(docid)
    with open("data/msmarco-docdev-top100") as f:
        for qid, _, docid, _, _, _ in csv.reader(f, delimiter=" "):
            if qid not in top100:
                top100[qid] = []
            top100[qid].append(docid)

    joblib.dump(querystring, "querystrings.pkl")
    joblib.dump(qrel, "qrels.pkl")
    joblib.dump(top100, "top100.pkl")
else:
    print("loading query-document metadata...")
    querystring = joblib.load("querystrings.pkl")
    qrel = joblib.load("qrels.pkl")
    top100 = joblib.load("top100.pkl")

preprocess = whoosh.analysis.StemmingAnalyzer()


def openfilesearcher():
    global FILE
    FILE = open("data/msmarco-docs.tsv", "rt", encoding="utf8")


def get_features(rank):
    features = []
    labels = []

    if rank == 0:
        pbar = tqdm(range(int(args.num_docs / 2 / 24)))
    else:
        pbar = range(int(args.num_docs / 2 / 24))
    for _ in pbar:
        try:
            qid = random.choice(list(querystring.keys()))
            query = [token.text for token in preprocess(querystring[qid])]

            # positive sample
            docid = random.choice(qrel[qid])
            docfeats = get_doc_feats(docid, query, FILE, SEARCHER)
            features.append(docfeats)
            if not args.graded:
                labels.append(1)
            else:
                labels.append(4 if random.random() < 0.5 else 3)

            # negative sample
            docid = random.choice(list(set(top100[qid]) - set(qrel[qid])))
            docfeats = get_doc_feats(docid, query, FILE, SEARCHER)
            features.append(docfeats)
            if not args.graded:
                labels.append(0)
            else:
                labels.append(1 if random.random() > 0.5 else 2)
        except Exception as e:
            print("ERROR:", e)

    return np.array(features), np.array(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-graded", action="store_true")
    parser.add_argument("-num_docs", type=int, default=10000)
    parser.add_argument("-num_graded", type=int, default=10000)
    args = parser.parse_args()

    print("opening searcher...")
    with FileStorage("data/msmarcoidx").open_index().searcher() as SEARCHER:

        print("getting features...")
        with mp.Pool(24, initializer=openfilesearcher) as pool:
            res = pool.imap(get_features, range(24))
            features, labels = [], []
            for f, l in res:
                features.append(f)
                labels.append(l)
            features = np.concatenate(features)
            labels = np.concatenate(labels)

        if args.graded:
            with open("graded-qrels.tsv", "rt", encoding="utf8") as gradqrelsfile:
                openfilesearcher()
                graded_features, graded_labels = [], []
                for qid, _, docid, relevance in tqdm(csv.reader(gradqrelsfile, delimiter=" "), total=316):
                    query = [token.text for token in preprocess(querystring[qid])]
                    docfeats = get_doc_feats(docid, query, FILE, SEARCHER)
                    graded_features.append(docfeats)
                    graded_labels.append(relevance)
                graded_features, graded_labels = np.array(graded_features), np.array(graded_labels)
                for _ in range(args.num_duplicate // 316):
                    features = np.concatenate((features, graded_features), axis=0)
                    labels = np.concatenate((labels, graded_labels), axis=0)

    print(features)
    print(features.shape, labels.shape)
    labels = labels.astype(np.float32)
    trainX, testX, trainY, testY = train_test_split(features, labels, test_size=0.1)

    print("training...")
    if args.graded:
        model = svm.SVR(max_iter=100_000).fit(trainX, trainY)
        print("Train MSE:", metrics.mean_squared_error(model.predict(trainX), trainY))
        testPred = model.predict(testX)
        [print(p, y) for p, y in zip(testPred, testY)]
        print("Test MSE:", metrics.mean_squared_error(testPred, testY))
        print("Test R^2:", metrics.r2_score(testPred, testY))
        print("Test MAE:", metrics.median_absolute_error(testPred, testY))
    else:
        model = svm.SVC(max_iter=100_000).fit(trainX, trainY)
        print("Train score:", np.mean(model.predict(trainX) == trainY))
        print("Test score:", np.mean(model.predict(testX) == testY))

    joblib.dump(model, "svm.pkl")
