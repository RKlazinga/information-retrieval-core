import os

import numpy as np
import pandas as pd
import whoosh.analysis
import whoosh.scoring
from tqdm import tqdm
from whoosh import index, writing
from whoosh.filedb.filestore import FileStorage
from sklearn import svm
from sklearn.model_selection import train_test_split
import joblib


def get_features():
    stem = whoosh.analysis.StemmingAnalyzer()

    triples = pd.read_csv(
        "data/triples.tsv",
        sep="\t",
        names=["topic", "query", "pos1", "pos2", "pos3", "pos4", "not1", "not2", "not3", "not4"],
    )

    ix = FileStorage("/HDDs/msmarco").open_index().reader()
    ndocs = ix.doc_count_all()
    avg_doc_len = ix.field_length("body") / ndocs

    features = np.array([[], [], [], [], [], [], []]).T
    labels = []
    idf, wfcorp = {}, {}

    for idx, (_, query, _, _, _, pos, _, _, _, neg) in tqdm(triples.iterrows()):
        try:
            query = [token.text for token in stem(query)]

            for i, doc in enumerate([neg] + [pos]):
                doc = [token.text for token in stem(doc)]
                intersection = set(query) & set(doc)
                docfeats = np.zeros(7)

                for term in intersection:
                    wfdoc = doc.count(term)
                    if term not in idf:
                        idf[term] = np.log(ndocs / (ix.doc_frequency("body", term) + 1e-8))
                        wfcorp[term] = ix.frequency("body", term) + 1e-8

                    docfeats[0] += np.log(wfdoc + 1)
                    docfeats[1] += np.log(idf[term])
                    docfeats[2] += np.log(wfdoc / len(doc) * idf[term] + 1)
                    docfeats[3] += np.log(ndocs / wfcorp[term] + 1)
                    docfeats[4] += np.log(wfdoc / len(doc) + 1)
                    docfeats[5] += np.log(wfdoc * ndocs / (len(doc) * wfcorp[term]) + 1)
                    docfeats[6] += whoosh.scoring.bm25(idf[term], wfcorp[term], len(doc), avg_doc_len, B=0.75, K1=1.2)

                if len(intersection) > 0:
                    docfeats[6] = np.log(docfeats[6])

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
