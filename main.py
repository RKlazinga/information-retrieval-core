import argparse
from okapi import Okapi
import os
import os.path

from whoosh import index, writing
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, KEYWORD, STORED, TEXT, Schema
from whoosh.filedb.filestore import FileStorage
from whoosh.qparser import QueryParser

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
parser.add_argument("-num_docs", type=int, default=None)
parser.add_argument("-query", type=str, default=None)
parser.add_argument("-reload", action="store_true")
args = parser.parse_args()

schema = Schema(
    docid=ID(stored=True),
    url=ID(stored=True),
    title=TEXT(stored=True, analyzer=StemmingAnalyzer()),  # maybe no stemming here?
    body=TEXT(analyzer=StemmingAnalyzer()),
)

if not os.path.exists("indexdir"):
    os.mkdir("indexdir")
    index.create_in("indexdir", schema)
    args.reload = True

storage = FileStorage("indexdir")
# Open an existing index
ix = storage.open_index()

if args.reload:
    ix.writer().commit(mergetype=writing.CLEAR)

    print(f"Loading documents from {args.data}")
    writer = ix.writer()
    with open(args.data, "r") as docs:
        i = 0
        line = docs.readline()
        while line != "" and (args.num_docs is None or i < args.num_docs):
            docid, url, title, body = line.split("\t")
            writer.add_document(docid=docid, url=url, title=title, body=body)
            line = docs.readline()
            i += 1
    writer.commit()

qp = QueryParser("body", schema=ix.schema)
q = qp.parse("Contents Hello")

from okapi import Okapi

with ix.searcher(weighting=Okapi) as s:
    results = s.search(q, terms=True)

    print(results)
    for res in results:
        print(res)

    # # Was this results object created with terms=True?
    # if results.has_matched_terms():
    #     # What terms matched in the results?
    #     print(results.matched_terms())

    #     # What terms matched in each hit?
    #     for hit in results:
    #         print(hit.matched_terms())
