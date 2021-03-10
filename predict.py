import argparse
import csv
import multiprocessing as mp

import joblib
import numpy as np
import whoosh
import whoosh.scoring
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from whoosh.analysis import StemmingAnalyzer
from whoosh.filedb.filestore import FileStorage
from whoosh.qparser import QueryParser

import okapi
import util
from features import get_doc_feats
from query_cluster import get_query_features
from train_cluster_svm import ClusterSVM

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, choices=["bm25", "okapi", "svc", "svr", "clusvm", "adarank"])
parser.add_argument("-svm_file", type=str, default="svm")
parser.add_argument("-limit", type=int, default=1000)
parser.add_argument("-binary", action="store_true")
parser.add_argument("-add_bm25", action="store_true")
args = parser.parse_args()

from singletons import PREPROCESS, SEARCHER

run_id = args.model
if args.model == "clusvm":
    clusvm = util.load(f"clusvm.pkl")
elif "sv" in args.model:
    svm = util.load(f"{args.svm_file}.pkl")
elif args.model == "adarank":
    alpha = np.load("ada.npy")
else:
    ix = FileStorage("data/msmarcoidx").open_index()
    if args.model == "bm25":
        SEARCHER = ix.searcher()
    qp = QueryParser("body", schema=ix.schema)


def predict(inp):
    qid, query = inp
    ret = []

    if args.model == "okapi" or args.model == "bm25":
        results = SEARCHER.search(qp.parse(query), limit=args.limit)
        for rank, hit in enumerate(results):
            ret.append([qid, hit["docid"], rank + 1, results.score(rank), run_id])

    elif args.model == "clusvm":
        query = [token.text for token in PREPROCESS(query)]
        queryfeats = get_query_features(query)
        queryfeats = np.concatenate([queryfeats[None, :]] * len(top100[qid]))

        docids = []
        features = []
        for docid in top100[qid]:
            features.append(get_doc_feats(docid, query, FILE))
            docids.append(docid)
        features = np.array(features)

        relevance = clusvm.predict(queryfeats, features)
        ordering = np.argsort(-relevance)

        for rank, idx in enumerate(ordering):
            if relevance[idx] != 0:
                ret.append([qid, docids[idx], rank + 1, relevance[idx], run_id])

    elif "sv" in args.model or args.model == "adarank":
        query = [token.text for token in PREPROCESS(query)]

        docids = []
        features = []
        for docid in top100[qid]:
            features.append(get_doc_feats(docid, query, FILE))
            docids.append(docid)
        features = np.array(features)

        if args.model == "adarank":
            relevance = np.dot(features, alpha)
        else:
            if args.binary or args.model == "svr":
                relevance = svm.predict(features)
            else:
                relevance = svm.decision_function(features)
            if args.add_bm25:
                relevance += features[:, -1] / 100
        ordering = np.argsort(-relevance)

        for rank, idx in enumerate(ordering):
            if relevance[idx] != 0:
                ret.append([qid, docids[idx], rank + 1, relevance[idx], run_id])

    else:
        print("ERROR: unsupported model")

    return ret


top100 = {}
with open("data/msmarco-doctest2019-top100") as f:
    for qid, _, docid, _, _, _ in csv.reader(f, delimiter=" "):
        if qid not in top100:
            top100[qid] = []
        top100[qid].append(docid)

# filter out the test queries which actually have qrels to evaluate with trec_eval
with open("data/2019qrels-docs.txt", "rt", encoding="utf8") as f:
    testedqueries = []
    for id, _, docid, rel in csv.reader(f, delimiter=" "):
        testedqueries.append(id)
with open("data/msmarco-test2019-queries.tsv", "rt", encoding="utf8") as f:
    queries = []
    for id, quer in csv.reader(f, delimiter="\t"):
        if id in testedqueries:
            queries.append([id, quer])


def opendocsfile():
    global FILE
    FILE = open("data/msmarco-docs.tsv", "rt", encoding="utf8")


print(f"Predicting {len(queries)} queries with {args.model}...")
querydocrankings = []
with tqdm(total=len(queries)) as pbar, mp.Pool(24, initializer=opendocsfile) as pool:
    for i, qdr in enumerate(pool.imap_unordered(predict, queries)):
        querydocrankings.append(qdr)
        pbar.update()

with open(f"output/{args.model}-predictions.trec", "w") as out:
    for querydocs in querydocrankings:
        for q, docid, r, score, run_id in querydocs:
            out.write(f"{q} Q0 {docid} {r} {score} {run_id}\n")
