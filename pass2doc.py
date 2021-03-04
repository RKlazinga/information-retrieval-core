import csv
import os

from tqdm import tqdm
from whoosh import index, writing
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.filedb.filestore import FileStorage
from whoosh.qparser import QueryParser

import okapi

need_reload = True

index_dir = "data/queryidx"
if not os.path.exists(index_dir):
    os.mkdir(index_dir)
    index.create_in(index_dir, Schema(qid=ID(stored=True), body=TEXT(stored=True)))
    need_reload = True

ix = FileStorage(index_dir).open_index()
if need_reload:
    ix.writer().commit(mergetype=writing.CLEAR)
    writer = ix.writer()
    with open(f"data/msmarco-doctrain-queries.tsv", "rt", encoding="utf8") as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, querystring_of_topicid] in tsvreader:
            writer.add_document(qid=topicid, body=querystring_of_topicid)
    with open(f"data/msmarco-docdev-queries.tsv", "rt", encoding="utf8") as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, querystring_of_topicid] in tsvreader:
            writer.add_document(qid=topicid, body=querystring_of_topicid)
    writer.commit()

queryix = FileStorage("data/queryidx").open_index()
docix = FileStorage("data/msmarcoidx").open_index()

qqp = QueryParser("body", schema=queryix.schema)
docqp = QueryParser("body", schema=docix.schema)

querytext = {}
with open("data/queries.train.tsv", "rt") as f:
    for qid, query in tqdm(csv.reader(f, delimiter="\t")):
        querytext[qid] = query
with open("data/queries.dev.tsv", "rt") as f:
    for qid, query in tqdm(csv.reader(f, delimiter="\t")):
        querytext[qid] = query

passagetext = {}
with open("data/collection.tsv", "rt") as f:
    for pid, text in tqdm(csv.reader(f, delimiter="\t")):
        passagetext[pid] = text

with open("data/thesis_dataset_graded_relevance.tsv", "rt") as grelfile, open(
    "data/pass2doc.json", "w"
) as out, docix.searcher(weighting=okapi.weighting) as s, queryix.searcher(weighting=okapi.weighting) as qs:
    for qid, pid, relevance in tqdm(csv.reader(grelfile, delimiter="\t"), total=800):
        closestqueries = qs.search(qqp.parse(querytext[qid]), limit=5)
        closestdocs = s.search(docqp.parse(passagetext[pid]), limit=5)
        if len(closestqueries) == 0 or len(closestdocs) == 0:
            print(f"No match found for {qid} {pid} {relevance}")
            print("queries", closestqueries)
            print("documents", closestdocs)
            continue
        for q, d in zip(closestqueries, closestdocs):
            print(f"{q['qid']} {d['docid']} {relevance}")
            out.write(f"{q['qid']} 0 {d['docid']} {relevance}")
