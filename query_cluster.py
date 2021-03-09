import random
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.cluster
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from singletons import PMI, PREPROCESS, QUERYDICT, SEARCHER


def get_query_features(query):
    ndocs = SEARCHER.reader().doc_count_all()

    feats = np.zeros((5, len(query)))

    for i, term in enumerate(query):
        df = SEARCHER.reader().doc_frequency("body", term)
        idf = np.log(ndocs / (df + 1e-8))
        wfcorp = SEARCHER.reader().frequency("body", term) + 1e-8

        feats[0, i] = idf
        feats[1, i] = wfcorp
        feats[2, i] = wfcorp * idf
        feats[3, i] = ndocs / wfcorp

        tpmi = 0
        for term2 in query:
            tpmi += PMI.compute(term, term2)
        feats[4, i] = tpmi

    feats = np.concatenate(
        (np.mean(feats, axis=1), np.max(feats, axis=1), np.sum(feats, axis=1), np.array([np.log(len(query) + 1)]))
    )
    feats = [np.log(score + 1e-8) if score > 0 else -np.log(-score + 1e-8) for score in feats]
    return np.array(feats)


def get_random_queryfeatures(k=20_000):
    with mp.Pool(24) as pool:
        querids = random.choices(list(QUERYDICT.keys()), k=k)
        queries = [[token.text for token in PREPROCESS(QUERYDICT[q])] for q in querids]
        return np.array(list(pool.imap_unordered(get_query_features, queries)))


if __name__ == "__main__":
    print("fitting DBSCANs...")
    label_counts = []
    for _ in tqdm(range(50)):
        label_counts.append(len(np.unique(sklearn.cluster.DBSCAN().fit_predict(get_random_queryfeatures()))))
    print("median / mean num labels:", int(np.median(label_counts)), int(np.mean(label_counts)))
    avg_num_labels = int(np.median(label_counts))
    127, 130, 128

    queryfeats = get_random_queryfeatures()

    train, test = train_test_split(queryfeats, test_size=0.25)
    cluster = sklearn.cluster.KMeans(n_clusters=avg_num_labels).fit(train)

    test_pred = cluster.predict(test)

    ks = [avg_num_labels]
    chs = [sklearn.metrics.calinski_harabasz_score(test, test_pred)]
    dbs = [sklearn.metrics.davies_bouldin_score(test, test_pred)]
    sil = [sklearn.metrics.silhouette_score(test, test_pred)]

    for n_clust in tqdm(range(10, 300, 10)):
        train, test = train_test_split(queryfeats, test_size=0.25)
        cluster = sklearn.cluster.KMeans(n_clusters=n_clust).fit(train)

        test_pred = cluster.predict(test)

        ks.append(n_clust)

        ch = sklearn.metrics.calinski_harabasz_score(test, test_pred)
        db = sklearn.metrics.davies_bouldin_score(test, test_pred)
        si = sklearn.metrics.silhouette_score(test, test_pred)

        print(n_clust, ch, db, si)

        chs.append(ch)
        dbs.append(db)
        sil.append(si)

    print(ks)
    print(chs)
    print(dbs)
    print(sil)
    indices = np.argsort(ks)
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.plot(np.array(ks)[indices], np.array(chs)[indices], label="calinski harabasz (higher better)", color=color)
    ax1.legend()
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.plot(np.array(ks)[indices], np.array(dbs)[indices], label="davies bouldin (lower better)", color=color)
    ax2.plot(np.array(ks)[indices], np.array(sil)[indices], label="silhouette (higher better)", color=color)
    ax2.legend()
    ax2.tick_params(axis="y", labelcolor=color)
    plt.tight_layout()
    plt.show()
