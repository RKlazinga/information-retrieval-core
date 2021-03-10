import csv
import os

import whoosh
from whoosh.filedb.filestore import FileStorage

import okapi
import util
from pmi import PMI

DOCOFFSET = {}
with open("data/msmarco-docs-lookup.tsv", "rt", encoding="utf8") as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [docid, _, offset] in tsvreader:
        DOCOFFSET[docid] = int(offset)


PREPROCESS = whoosh.analysis.StemmingAnalyzer()


if not os.path.exists("querystrings.pkl"):
    # if True:
    # The query string for each topicid is querystring[topicid]
    QUERYDICT = {}
    with open("data/msmarco-doctrain-queries.tsv", "rt", encoding="utf8") as f:
        for [topicid, querystring_of_topicid] in csv.reader(f, delimiter="\t"):
            QUERYDICT[topicid] = querystring_of_topicid
    with open("data/msmarco-docdev-queries.tsv", "rt", encoding="utf8") as f:
        for [topicid, querystring_of_topicid] in csv.reader(f, delimiter="\t"):
            QUERYDICT[topicid] = querystring_of_topicid

    # For each topicid, the list of positive docids is qrel[topicid]
    QREL = {}
    with open("data/msmarco-doctrain-qrels.tsv", "rt", encoding="utf8") as f:
        for [topicid, _, docid, rel] in csv.reader(f, delimiter=" "):
            assert rel == "1"
            if topicid in QREL:
                QREL[topicid].append(docid)
            else:
                QREL[topicid] = [docid]
    with open("data/msmarco-docdev-qrels.tsv", "rt", encoding="utf8") as f:
        for [topicid, _, docid, rel] in csv.reader(f, delimiter=" "):
            assert rel == "1"
            if topicid in QREL:
                QREL[topicid].append(docid)
            else:
                QREL[topicid] = [docid]

    TOP100 = {}
    with open("data/msmarco-doctrain-top100") as f:
        for qid, _, docid, _, _, _ in csv.reader(f, delimiter=" "):
            if qid not in TOP100:
                TOP100[qid] = []
            TOP100[qid].append(docid)
    with open("data/msmarco-docdev-top100") as f:
        for qid, _, docid, _, _, _ in csv.reader(f, delimiter=" "):
            if qid not in TOP100:
                TOP100[qid] = []
            TOP100[qid].append(docid)

    util.save("querystrings.pkl", QUERYDICT)
    util.save("qrels.pkl", QREL)
    util.save("top100.pkl", TOP100)
else:
    QUERYDICT = util.load("querystrings.pkl")
    QREL = util.load("qrels.pkl")
    TOP100 = util.load("top100.pkl")


SEARCHER = FileStorage("data/msmarcoidx").open_index().searcher(weighting=okapi.weighting)


PMI = util.load("pmi1m.pkl")
