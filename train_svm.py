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
from singletons import TOP100, QUERYDICT, PREPROCESS, QREL


def get_features(inp):
    features = []
    labels = []

    for _ in inp:
        try:
            qid = random.choice(list(QUERYDICT.keys()))
            query = [token.text for token in PREPROCESS(QUERYDICT[qid])]

            # positive sample
            docid = random.choice(QREL[qid])
            docfeats = get_doc_feats(docid, query, FILE)
            features.append(docfeats)
            if not args.graded:
                labels.append(1)
            else:
                labels.append(4 if random.random() < 0.5 else 3)

            # negative sample
            docid = random.choice(list(set(TOP100[qid]) - set(QREL[qid])))
            docfeats = get_doc_feats(docid, query, FILE)
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
    query = [token.text for token in PREPROCESS(QUERYDICT[qid])]
    docfeats = get_doc_feats(docid, query, FILE)
    return docfeats, relevance


def opendocsfile():
    global FILE
    FILE = open("data/msmarco-docs.tsv", "rt", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-graded", action="store_true")
    parser.add_argument("-num_docs", type=int, default=25000)
    parser.add_argument("-num_graded", type=int, default=25000)
    parser.add_argument("-load", type=str, default=None)
    args = parser.parse_args()

    if args.load is None:
        print("getting features...")
        features, labels = [], []
        pbar = tqdm(total=args.num_docs)
        chunksize = 10
        with mp.Pool(24, initializer=opendocsfile) as pool:
            for f, l in pool.imap_unordered(get_features, util.chunks(range(int(args.num_docs / 2)), n=chunksize)):
                features.append(f.squeeze())
                labels.append(l.squeeze())
                pbar.update(chunksize * 2)
            features = np.concatenate(features)
            labels = np.concatenate(labels)

        if args.graded:
            with open("graded-qrels.tsv", "rt", encoding="utf8") as gradqrelsfile:
                pbar = tqdm(total=316)
                graded_features, graded_labels = [], []
                with mp.Pool(24) as pool:
                    for docfeats, relevance in pool.imap_unordered(
                        process_graded, csv.reader(gradqrelsfile, delimiter=" ")
                    ):
                        graded_features.append(docfeats)
                        graded_labels.append(relevance)
                        pbar.update()
            for _ in range(args.num_graded // 316):
                features = np.concatenate((features, graded_features), axis=0)
                labels = np.concatenate((labels, graded_labels), axis=0)
            labels = labels.astype(np.float32)

        util.save(
            f"doc_features_{args.num_docs + (args.num_graded if args.graded else 0)}_{'graded' if args.graded else 'binary'}.pkl",
            (features, labels),
        )
    else:
        features, labels = util.load(args.load)
    features[~np.isfinite(features)] = 0

    print(features)
    print(features.shape, labels.shape)

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
