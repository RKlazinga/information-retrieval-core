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

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
parser.add_argument("model", type=str)
parser.add_argument("-num_docs", type=int, default=None)
parser.add_argument("-threads", type=int, default=128)
parser.add_argument("-reload", action="store_true")
parser.add_argument("-interactive", action="store_true")
parser.add_argument("-query", type=str, default=None)
args = parser.parse_args()

schema = Schema(
    docid=ID(stored=True),
    url=ID(stored=True),
    title=TEXT(stored=True, analyzer=StemmingAnalyzer()),  # maybe no stemming here?
    body=TEXT(analyzer=StemmingAnalyzer()),
)

index_dir = "/HDDs/msmarco" if args.num_docs is None else "data/quickidx"
if not os.path.exists(index_dir):
    os.mkdir(index_dir)
    index.create_in(index_dir, schema)
    args.reload = True

storage = FileStorage(index_dir)
# Open an existing index
ix = storage.open_index()

if args.reload:
    ix.writer().commit(mergetype=writing.CLEAR)

    print(f"Loading documents from {args.data}")
    writers = [AsyncWriter(ix) for _ in range(args.threads)]
    with open(args.data, "r", encoding="utf-8") as docs:
        i = 0
        line = docs.readline()
        pbar = tqdm(total=args.num_docs if args.num_docs is not None else 3_213_835)
        while line != "" and (args.num_docs is None or i < args.num_docs):
            docid, url, title, body = line.split("\t")
            writers[i % args.threads].add_document(docid=docid, url=url, title=title, body=body)
            line = docs.readline()
            i += 1
            pbar.update(1)
    pbar.set_description("Committing...")
    pbar.refresh()

    [w.commit() for w in writers]

# In the corpus tsv, each docid occurs at offset docoffset[docid]
docoffset = {}
with open("data/msmarco-docs-lookup.tsv", "rt", encoding="utf8") as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [docid, _, offset] in tsvreader:
        docoffset[docid] = int(offset)


def getbody(docid, f):
    f.seek(docoffset[docid])
    line = f.readline()
    assert line.startswith(docid + "\t"), f"Looking for {docid}, found {line}"
    linelist = line.rstrip().split("\t")
    if len(linelist) == 4:
        return linelist[3]


qp = QueryParser("body", schema=ix.schema)

if args.interactive:

    def interactive_search(query):
        q = qp.parse(query)

        print("OKAPI")
        with ix.searcher(weighting=okapi.weighting) as s:
            results = s.search(q, terms=True)
            for hit in results:
                print(hit, hit.matched_terms())

        print("\nBM25F")
        with ix.searcher() as s:
            results = s.search(q, terms=True)
            for hit in results:
                print(hit, hit.matched_terms())

    if args.query is not None:
        interactive_search(args.query)
    else:
        while True:
            query = input("query: ")
            interactive_search(query)
    exit(0)

run_id = args.model
if args.model == "svm":
    svm = joblib.load("svm.pkl")
elif args.model == "adarank":
    alpha = np.load("ada.npy")

ndocs = ix.doc_count_all()
avg_doc_len = ix.field_length("body") / ndocs
idf, wfcorp = {}, {}
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
        results = s.search(qp.parse(query), limit=100)
        query = [token.text for token in stem(query)]

        docids = []
        for rank, hit in enumerate(results):
            docids.append(hit["docid"])
            body = getbody(hit["docid"], docs)
            body = [token.text for token in stem(body)]
            intersection = set(query) & set(body)
            docfeats = np.zeros(7, dtype=np.float64)
            for term in intersection:
                wfdoc = body.count(term)
                if term not in idf:
                    idf[term] = np.log(ndocs / (r.doc_frequency("body", term) + 1e-8))
                    wfcorp[term] = r.frequency("body", term) + 1e-8

                docfeats[0] += np.log(wfdoc + 1)
                docfeats[1] += np.log(idf[term])
                docfeats[2] += np.log(wfdoc / len(body) * idf[term] + 1)
                docfeats[3] += np.log(ndocs / wfcorp[term] + 1)
                docfeats[4] += np.log(wfdoc / len(body) + 1)
                docfeats[5] += np.log(wfdoc * ndocs / (len(body) * wfcorp[term]) + 1)
                docfeats[6] += whoosh.scoring.bm25(idf[term], wfcorp[term], len(body), avg_doc_len, B=0.75, K1=1.2)

            if len(intersection) > 0:
                docfeats[6] = np.log(docfeats[6])

            features.append(docfeats)
        features = np.array(features)

        try:
            if args.model == "adarank":
                relevance = np.dot(features, alpha)
                ordering = np.argsort(relevance)
            else:
                relevance = svm.predict(features)
                ordering = np.argsort(relevance)

            for rank, idx in enumerate(ordering[:25]):
                if relevance[idx] != 0:
                    ret.append([qid, docids[idx], rank + 1, relevance[idx], run_id])
        except:
            print(qid, query, features)

    return ret


with open("data/msmarco-test2019-queries.tsv", "rt", encoding="utf8") as f, open(
    f"{args.model}-predictions.trec", "w"
) as out, open("data/msmarco-docs.tsv", "rt", encoding="utf8") as docs:

    queries = []
    for qid, query in csv.reader(f, delimiter="\t"):
        queries.append([qid, query])

    print(f"Prediction {len(queries)} queries with {args.model}...")
    with ix.searcher(weighting=okapi.weighting if args.model == "okapi" else whoosh.scoring.BM25F) as s:
        r = ix.reader()
        querydocrankings = []
        for query in tqdm(queries):
            querydocrankings.append(predict(query))

    for querydocs in querydocrankings:
        for qid, docid, rank, score, run_id in querydocs:
            out.write(f"{qid} Q0 {docid} {rank} {score} {run_id}\n")
