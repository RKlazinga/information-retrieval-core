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
parser.add_argument("-preselection", type=int, default=50)
args = parser.parse_args()

ix = FileStorage("data/msmarcoidx").open_index()
qp = QueryParser("body", schema=ix.schema)

run_id = args.model
if args.model == "svm":
    svm = joblib.load("svm.pkl")
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
                relevance = svm.predict(features) + features[:, -1] / 100
            ordering = np.argsort(-relevance)

            for rank, idx in enumerate(ordering[:25]):
                ret.append([qid, docids[idx], rank + 1, relevance[idx], run_id])

        except:
            print("ERROR", qid, query, features)

    return ret


with open("data/msmarco-test2019-queries.tsv", "rt", encoding="utf8") as f:
    queries = []
    for qi, quer in csv.reader(f, delimiter="\t"):
        queries.append([qi, quer])

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
