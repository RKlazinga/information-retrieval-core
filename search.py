import argparse
from whoosh.filedb.filestore import FileStorage
from whoosh.qparser import QueryParser

import okapi

parser = argparse.ArgumentParser()
parser.add_argument("-query", type=str, default=None)
args = parser.parse_args()

ix = FileStorage("data/msmarcoidx").open_index()
qp = QueryParser("body", schema=ix.schema)


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


args.query = "shipping"
print(args.query)
if args.query is not None:
    interactive_search(args.query)
else:
    while True:
        query = input("query: ")
        interactive_search(query)
