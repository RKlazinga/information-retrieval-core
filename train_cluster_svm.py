import argparse
import csv
import multiprocessing as mp
import os

import numpy as np
import sklearn.cluster
import sklearn.metrics as metrics
import sklearn.svm
import whoosh.analysis
import whoosh.scoring
from sklearn.model_selection import train_test_split
from whoosh.filedb.filestore import FileStorage

import util
from query_cluster import get_random_queryfeatures


class ClusterSVM:
    def __init__(self, qf, df, labels, graded=False):
        self.graded = graded
        self.cluster = sklearn.cluster.DBSCAN().fit(qf)
        predictions = self.cluster.predict(qf)

        self.svms = [sklearn.svm.SVR() if graded else sklearn.svm.SVC()]
        for l, svm in enumerate(self.svms):
            indices = predictions == l
            svm.fit(df[indices], labels[indices])

    def predict(self, qf, df):
        labels = self.cluster.predict(qf)
        preds = []
        for l in labels:
            preds.append(self.svms[l].predict(df) if self.graded else self.svms[l].decision_function(df))
        return np.array(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-graded", action="store_true")
    parser.add_argument("-num_docs", type=int, default=25000)
    parser.add_argument("-num_graded", type=int, default=25000)
    parser.add_argument("-load", type=str, default="document_features_25000_binary.pkl")
    args = parser.parse_args()

    features = util.load(args.load)
    labels = util.load(args.load.replace("features", "labels"))

    print(np.sum(~np.isfinite(features)))
    features[~np.isfinite(features)] = 0
    print(features)
    print(features.shape, labels.shape)

    queryfeats = get_random_queryfeatures(k=100_000)

    trainX, testX, trainY, testY = train_test_split(features, labels, test_size=0.1)
    trainQ, testQ = train_test_split(queryfeats, test_size=0.1)

    print("training...")
    clusvm = ClusterSVM(trainQ, trainX, trainY, args.graded)
    trainPred = clusvm.predict(testQ, testX)
    testPred = clusvm.predict(testQ, testX)

    if args.graded:
        print("Train MSE:", metrics.mean_squared_error(trainPred, trainY))
        print("Test MSE:", metrics.mean_squared_error(testPred, testY))
        print("Test R^2:", metrics.r2_score(testPred, testY))
        print("Test MAE:", metrics.median_absolute_error(testPred, testY))
    else:
        print("Train score:", np.mean(trainPred == trainY))
        print("Test score:", np.mean(testPred == testY))

    util.save("clusvm.pkl", clusvm)
