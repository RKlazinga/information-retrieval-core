import csv
import multiprocessing as mp
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


def retrieve(inp):
    qid, pid, relevance = inp
    ret = []
    closestqueries = qs.search(qqp.parse(querytext[qid]), limit=5)
    closestdocs = s.search(docqp.parse(passagetext[pid]), limit=5)
    if len(closestqueries) == 0 or len(closestdocs) == 0:
        return None
    for q, d in zip(closestqueries, closestdocs):
        ret.append([q["qid"], d["docid"], relevance])
    return ret


qrels = []
with open("data/thesis_dataset_graded_relevance.tsv", "rt") as grelfile, docix.searcher(
    weighting=okapi.weighting
) as s, queryix.searcher(weighting=okapi.weighting) as qs, open("data/graded-qrels.txt", "w") as out:
    with mp.Pool(processes=24) as pool:
        with tqdm(total=800) as pbar:
            found, sofar = 0, 0
            for passdoc in pool.imap_unordered(retrieve, csv.reader(grelfile, delimiter="\t")):
                sofar += 1
                pbar.set_description(f"{found} / {sofar}")
                pbar.update()
                if passdoc is None:
                    continue
                found += 1
                pbar.set_description(f"{found} / {sofar}")
                for pd in passdoc:
                    qrels.append(pd)
                    out.write(f"{pd[0]} 0 {pd[1]} {pd[2]}")