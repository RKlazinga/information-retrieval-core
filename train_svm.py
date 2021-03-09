import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import time

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

import util
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

    util.save("querystrings.pkl", querystring)
    util.save("qrels.pkl", qrel)
    util.save("top100.pkl", top100)
else:
    querystring = util.load("querystrings.pkl")
    qrel = util.load("qrels.pkl")
    top100 = util.load("top100.pkl")

preprocess = whoosh.analysis.StemmingAnalyzer()


def openfilesearcher():
    global FILE
    FILE = open("data/msmarco-docs.tsv", "rt", encoding="utf8")


def get_features(inp):
    features = []
    labels = []

    for _ in inp:
        try:
            qid = random.choice(list(querystring.keys()))
            query = [token.text for token in preprocess(querystring[qid])]

            # positive sample
            docid = random.choice(qrel[qid])
            docfeats = get_doc_feats(docid, query, FILE, SEARCHER, PMI1M)
            features.append(docfeats)
            if not args.graded:
                labels.append(1)
            else:
                labels.append(4 if random.random() < 0.5 else 3)

            # negative sample
            docid = random.choice(list(set(top100[qid]) - set(qrel[qid])))
            docfeats = get_doc_feats(docid, query, FILE, SEARCHER, PMI1M)
            features.append(docfeats)
            if not args.graded:
                labels.append(0)
            else:
                labels.append(1 if random.random() > 0.5 else 2)
        except Exception as e:
            print("ERROR:", e)

    return np.array(features), np.array(labels)


def process_graded(inp):
    qid, _, docid, relevance = inp
    query = [token.text for token in preprocess(querystring[qid])]
    docfeats = get_doc_feats(docid, query, FILE, SEARCHER, PMI1M)
    return docfeats, relevance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-graded", action="store_true")
    parser.add_argument("-num_docs", type=int, default=10000)
    parser.add_argument("-num_graded", type=int, default=10000)
    args = parser.parse_args()

    print("opening searcher...")
    tik = time.time()
    with FileStorage("data/msmarcoidx").open_index().searcher() as SEARCHER:
        print("took:", time.time() - tik)
        PMI1M = util.load("pmi1m.pkl")

        print("getting features...")
        features, labels = [], []
        with mp.Pool(24, initializer=openfilesearcher) as pool:
            pbar = tqdm(total=args.num_docs)
            chunksize = 10  # int(args.num_docs / 24 / 16)
            for f, l in pool.imap_unordered(get_features, util.chunks(range(int(args.num_docs / 2)), n=chunksize)):
                features.append(f.squeeze())
                labels.append(l.squeeze())
                pbar.update(chunksize)
            features = np.concatenate(features)
            labels = np.concatenate(labels)

            if args.graded:
                with open("graded-qrels.tsv", "rt", encoding="utf8") as gradqrelsfile:
                    pbar = tqdm(total=316)
                    graded_features, graded_labels = [], []
                    for docfeats, relevance in pool.imap_unordered(
                        process_graded, csv.reader(gradqrelsfile, delimiter=" ")
                    ):
                        graded_features.append(docfeats)
                        graded_labels.append(relevance)
                        pbar.update()
                for _ in range(args.num_graded // 316):
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

    util.save("svm.pkl", model)
