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
from util import features_per_doc, getbody

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, choices=["bm25", "okapi", "svm", "adarank"])
parser.add_argument("-svm_file", type=str, choices=["svm1k", "svm100k"], default="svm100k")
parser.add_argument("-preselection", type=int, default=50)
parser.add_argument("-binary", action="store_true")
parser.add_argument("-add_bm25", action="store_true")
args = parser.parse_args()

ix = FileStorage("data/msmarcoidx").open_index()
qp = QueryParser("body", schema=ix.schema)

run_id = args.model
if args.model == "svm":
    svm = joblib.load(f"{args.svm_file}.pkl")
elif args.model == "adarank":
    alpha = np.load("ada.npy")

stem = StemmingAnalyzer()


def openfilesearcher():
    global DOCSFILE
    DOCSFILE = open("data/msmarco-docs.tsv", "rt", encoding="utf8")


def predict(inp):
    global DOCSFILE
    qid, query = inp
    ret = []

    if args.model == "okapi" or args.model == "bm25":
        results = SEARCHER.search(qp.parse(query), limit=25)
        for rank, hit in enumerate(results):
            ret.append([qid, hit["docid"], rank + 1, results.score(rank), run_id])

    if args.model == "svm" or args.model == "adarank":
        results = SEARCHER.search(qp.parse(query), limit=args.preselection)
        query = [token.text for token in stem(query)]

        docids = []
        features = []
        for rank, hit in enumerate(results):
            body = [token.text for token in stem(getbody(hit["docid"], DOCSFILE))]

            features.append(features_per_doc(query, body, SEARCHER))
            docids.append(hit["docid"])
        features = np.array(features)

        try:
            if args.model == "adarank":
                relevance = np.dot(features, alpha)
            else:
                if args.binary:
                    relevance = svm.predict(features)
                else:
                    relevance = svm.decision_function(features)
                if args.add_bm25:
                    relevance += features[:, -1] / 100
            ordering = np.argsort(-relevance)

            for rank, idx in enumerate(ordering[:25]):
                if relevance[idx] != 0:
                    ret.append([qid, docids[idx], rank + 1, relevance[idx], run_id])

        except:
            print("ERROR: query ", qid, query, " has no results ", results, features)

    return ret


# filter out the test queries which actually have qrels to evaluate with trec_eval
with open("2019qrels-docs.txt", "rt", encoding="utf8") as f:
    testedqueries = []
    for id, quer in csv.reader(f, delimiter="\t"):
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
