import argparse
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
parser.add_argument("-query", type=str, default=None)
args = parser.parse_args()

schema = Schema(
    docid=ID(stored=True),
    url=ID(stored=True),
    title=TEXT(stored=True, analyzer=StemmingAnalyzer()),  # maybe no stemming here?
    body=TEXT(analyzer=StemmingAnalyzer()),
)

if not os.path.exists("indexdir" if args.num_docs is None else "l2r"):
    os.mkdir("indexdir" if args.num_docs is None else "l2r")
    index.create_in("indexdir" if args.num_docs is None else "l2r", schema)
    args.reload = True

storage = FileStorage("indexdir" if args.num_docs is None else "l2r")
# Open an existing index
ix = storage.open_index()

if args.reload:
    ix.writer().commit(mergetype=writing.CLEAR)

    print(f"Loading documents from {args.data}")
    writer = ix.writer()
    with open(args.data, "r") as docs:
        i = 0
        line = docs.readline()
        pbar = tqdm(total=args.num_docs if args.num_docs is not None else 3213835)
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
        if results.has_matched_terms():
            for hit in results:
                print(hit, hit.matched_terms())

    print("\nBM25F")
    with ix.searcher() as s:
        results = s.search(q, terms=True)
        if results.has_matched_terms():
            for hit in results:
                print(hit, hit.matched_terms())


qp = QueryParser("body", schema=ix.schema)
if args.query is not None:
    search(args.query)
else:
    while True:
        query = input("query: ")
        search(query)

