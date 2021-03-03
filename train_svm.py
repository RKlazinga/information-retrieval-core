import os

import numpy as np
import pandas as pd
import whoosh.analysis
import whoosh.scoring
from tqdm import tqdm
from whoosh.filedb.filestore import FileStorage
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib
from util import features_per_doc


def get_features():
    stem = whoosh.analysis.StemmingAnalyzer()
    triples = pd.read_csv(
        "data/triples.tsv",
        sep="\t",
        names=["topic", "query", "pos1", "pos2", "pos3", "pos4", "not1", "not2", "not3", "not4"],
    )
    ix = FileStorage("/HDDs/msmarco").open_index().reader()

    features = np.array([[], [], [], [], [], [], []]).T
    labels = []

    for _, (_, query, _, _, _, pos, _, _, _, neg) in tqdm(triples.iterrows()):
        try:
            query = [token.text for token in stem(query)]
            for i, doc in enumerate([neg] + [pos]):
                doc = [token.text for token in stem(doc)]
                docfeats = features_per_doc(query, doc, ix)
                features = np.concatenate([features, docfeats[None, :]], axis=0)
                labels.append(i)

        except:
            print("\n\n\nERRROR")
            print("query", query)
            print("pos", pos)
            print("neg", neg)

    return features, np.array(labels)


if __name__ == "__main__":
    features, labels = get_features()

    trainX, testX, trainY, testY = train_test_split(features, labels, test_size=0.2)

    model = svm.SVC(max_iter=100_000).fit(trainX, trainY)

    print("Train score:", np.mean(model.predict(trainX) == trainY))
    print("Test score:", np.mean(model.predict(testX) == testY))

    joblib.dump(model, "svm.pkl")
