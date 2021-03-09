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
from features import get_doc_feats
from pmi import PMI

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, choices=["bm25", "okapi", "svc", "svr", "adarank"])
parser.add_argument("-svm_file", type=str, default="svm")
parser.add_argument("-limit", type=int, default=1000)
parser.add_argument("-binary", action="store_true")
parser.add_argument("-add_bm25", action="store_true")
args = parser.parse_args()

ix = FileStorage("data/msmarcoidx").open_index()
qp = QueryParser("body", schema=ix.schema)

run_id = args.model
if "sv" in args.model:
    svm = joblib.load(f"{args.svm_file}.pkl")
elif args.model == "adarank":
    alpha = np.load("ada.npy")

top100 = {}
with open("data/msmarco-doctest2019-top100") as f:
    for qid, _, docid, _, _, _ in csv.reader(f, delimiter=" "):
        if qid not in top100:
            top100[qid] = []
        top100[qid].append(docid)

preprocess = whoosh.analysis.StemmingAnalyzer()


def openfilesearcher():
    global FILE
    FILE = open("data/msmarco-docs.tsv", "rt", encoding="utf8")


def predict(inp):
    qid, query = inp
    ret = []

    if args.model == "okapi" or args.model == "bm25":
        results = SEARCHER.search(qp.parse(query), limit=args.limit)
        for rank, hit in enumerate(results):
            ret.append([qid, hit["docid"], rank + 1, results.score(rank), run_id])

    if "sv" in args.model or args.model == "adarank":
        query = [token.text for token in preprocess(query)]

        docids = []
        features = []
        for docid in top100[qid]:
            features.append(get_doc_feats(docid, query, FILE, SEARCHER))
            docids.append(docid)
        features = np.array(features)

        # try:
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
        # except Exception as e:
        #     print("ERROR:", e)

    return ret


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

print(f"Predicting {len(queries)} queries with {args.model}...")
querydocrankings = []
with ix.searcher(weighting=whoosh.scoring.BM25F if args.model == "bm25" else okapi.weighting) as SEARCHER:
    with mp.Pool(processes=24, initializer=openfilesearcher) as pool:
        with tqdm(total=len(queries)) as pbar:
            for i, qdr in enumerate(pool.imap_unordered(predict, queries)):
                querydocrankings.append(qdr)
                pbar.update()

with open(f"output/{args.model}-predictions.trec", "w") as out:
    for querydocs in querydocrankings:
        for q, docid, r, score, run_id in querydocs:
            out.write(f"{q} Q0 {docid} {r} {score} {run_id}\n")
