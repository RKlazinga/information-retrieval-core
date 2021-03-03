import csv
from typing import List

import numpy as np
import whoosh.scoring

IDF, WFCORP = {}, {}  # caches
NDOCS = None
AVG_DOC_LEN = None


def features_per_doc(query: List, doc: List, reader):
    global NDOCS, AVG_DOC_LEN
    if NDOCS is None:
        NDOCS = reader.doc_count_all()
        AVG_DOC_LEN = reader.field_length("body") / NDOCS

    intersection = set(query) & set(doc)
    docfeats = np.zeros(7)

    for term in intersection:
        wfdoc = doc.count(term)
        if term not in IDF:
            IDF[term] = np.log(NDOCS / (reader.doc_frequency("body", term) + 1e-8))
            WFCORP[term] = reader.frequency("body", term) + 1e-8

        docfeats[0] += np.log(wfdoc + 1)
        docfeats[1] += np.log(IDF[term])
        docfeats[2] += np.log(wfdoc / len(doc) * IDF[term] + 1)
        docfeats[3] += np.log(NDOCS / WFCORP[term] + 1)
        docfeats[4] += np.log(wfdoc / len(doc) + 1)
        docfeats[5] += np.log(wfdoc * NDOCS / (len(doc) * WFCORP[term]) + 1)
        docfeats[6] += whoosh.scoring.bm25(IDF[term], WFCORP[term], len(doc), AVG_DOC_LEN, B=0.75, K1=1.2)

    if len(intersection) > 0:
        docfeats[6] = np.log(docfeats[6])

    return docfeats


DOCOFFSET = None


def getbody(docid, docsfile):
    global DOCOFFSET
    if DOCOFFSET is None:
        DOCOFFSET = {}
        with open("data/msmarco-docs-lookup.tsv", "rt", encoding="utf8") as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [docid, _, offset] in tsvreader:
                DOCOFFSET[docid] = int(offset)

    docsfile.seek(DOCOFFSET[docid])
    line = docsfile.readline()
    assert line.startswith(docid + "\t"), f"Looking for {docid}, found {line}"
    linelist = line.rstrip().split("\t")
    if len(linelist) == 4:
        return linelist[3]
