import argparse
import csv
import os.path

import joblib
import numpy as np
import whoosh
import whoosh.scoring
from tqdm import tqdm
from whoosh import index, writing
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.filedb.filestore import FileStorage
from whoosh.qparser import QueryParser
from whoosh.writing import AsyncWriter

import okapi
from util import features_per_doc, getbody

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
args = parser.parse_args()

ix = FileStorage("/HDDs/msmarco").open_index()
qp = QueryParser("body", schema=ix.schema)

run_id = args.model
if args.model == "svm":
    svm = joblib.load("svm.pkl")
elif args.model == "adarank":
    alpha = np.load("ada.npy")

stem = StemmingAnalyzer()


def predict(inp):
    qid, query = inp
    ret = []

    if args.model == "okapi" or args.model == "bm25":
        results = s.search(qp.parse(query), limit=25)
        for rank, hit in enumerate(results):
            ret.append(qid, hit["docid"], rank + 1, results.score(rank), run_id)

    if args.model == "svm" or args.model == "adarank":
        features = []
        results = s.search(qp.parse(query), limit=1000)
        query = [token.text for token in stem(query)]

        docids = []
        for rank, hit in enumerate(results):
            docids.append(hit["docid"])

            body = [token.text for token in stem(getbody(hit["docid"], docs))]

            docfeats = features_per_doc(query, body, r)

            features.append(docfeats)
        features = np.array(features)

        try:
            if args.model == "adarank":
                relevance = np.dot(features, alpha)
                ordering = np.argsort(relevance)
            else:
                relevance = svm.predict(features) + features[:, -1] / 100
                ordering = np.argsort(relevance)

            for rank, idx in enumerate(ordering[:25]):
                if relevance[idx] != 0:
                    ret.append([qid, docids[idx], rank + 1, relevance[idx], run_id])
        except:
            print(qid, query, features)

    print(ret)
    return ret


with open("data/msmarco-test2019-queries.tsv", "rt", encoding="utf8") as f, open(
    f"{args.model}-predictions.trec", "w"
) as out, open("data/msmarco-docs.tsv", "rt", encoding="utf8") as docs:

    queries = []
    for qid, query in csv.reader(f, delimiter="\t"):
        queries.append([qid, query])

    print(f"Predicting {len(queries)} queries with {args.model}...")
    with ix.searcher(weighting=okapi.weighting if args.model == "okapi" else whoosh.scoring.BM25F) as s:
        r = ix.reader()
        querydocrankings = []
        for query in tqdm(queries):
            querydocrankings.append(predict(query))

    for querydocs in querydocrankings:
        for qid, docid, rank, score, run_id in querydocs:
            out.write(f"{qid} Q0 {docid} {rank} {score} {run_id}\n")
