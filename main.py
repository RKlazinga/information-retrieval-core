import argparse
import csv
import os
import os.path

from tqdm import tqdm
from whoosh import index, writing
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.filedb.filestore import FileStorage
from whoosh.qparser import QueryParser

import okapi

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
parser.add_argument("-num_docs", type=int, default=None)
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
    writer = ix.writer()
    with open(args.data, "r") as docs:
        i = 0
        line = docs.readline()
        pbar = tqdm(total=args.num_docs if args.num_docs is not None else 3_213_835)
        while line != "" and (args.num_docs is None or i < args.num_docs):
            docid, url, title, body = line.split("\t")
            writer.add_document(docid=docid, url=url, title=title, body=body)
            line = docs.readline()
            i += 1
            pbar.update(1)
    pbar.set_description("Committing...")
    pbar.refresh()
    writer.commit()


def search(query):
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


qp = QueryParser("body", schema=ix.schema)

if args.interactive:
    if args.query is not None:
        search(args.query)
    else:
        while True:
            query = input("query: ")
            search(query)
else:
    run_id = "okapi"
    pbar = tqdm(total=200)
    with open("data/msmarco-test2019-queries.tsv", "rt", encoding="utf8") as f, open(
        "okapi-predictions.trec", "w"
    ) as out:
        for qid, query in csv.reader(f, delimiter="\t"):
            with ix.searcher(weighting=okapi.weighting) as s:
                results = s.search(qp.parse(query), limit=25)
                for rank, hit in enumerate(results):
                    out.write(f"{qid} Q0 {hit['docid']} {rank + 1} {results.score(rank)} {run_id}\n")
            pbar.update(1)
