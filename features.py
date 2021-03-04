import csv
from typing import List

import numpy as np
import whoosh.scoring

U_DIR = 2000
LMD_SHORT = 0.65
LMD_LONG = 0.25

preprocess = whoosh.analysis.StemmingAnalyzer()


def get_doc_feats(docid, query, FILE, SEARCHER, use_title=True):
    body = getbody(docid, FILE)
    doc = [token.text for token in preprocess(body)]
    docfeats = features_per_doc(query, doc, SEARCHER)
    if not use_title:
        return docfeats

    title = getbody(docid, FILE, index=2)
    title = [token.text for token in preprocess(title)]
    titlefeats = features_per_doc(query, title, SEARCHER, field="title")

    return np.concatenate((docfeats, titlefeats))


def features_per_doc(query: List, doc: List, searcher, field="body"):
    CORP_LEN = searcher.reader().field_length(field)
    NDOCS = searcher.reader().doc_count_all()
    AVG_DOC_LEN = CORP_LEN / NDOCS

    intersection = set(query) & set(doc)
    docfeats = np.ones(10)

    # AdaRank features
    for term in intersection:
        wfdoc = doc.count(term)
        idf = np.log(NDOCS / (searcher.reader().doc_frequency(field, term) + 1e-8))
        wfcorp = searcher.reader().frequency(field, term) + 1e-8

        docfeats[0] += wfdoc + 1
        docfeats[1] += idf
        docfeats[2] += wfdoc / len(doc) * idf + 1
        docfeats[3] += NDOCS / wfcorp + 1
        docfeats[4] += wfdoc / len(doc) + 1
        docfeats[5] += wfdoc * NDOCS / (len(doc) * wfcorp) + 1
        docfeats[6] += whoosh.scoring.bm25(idf, wfcorp, len(doc), AVG_DOC_LEN, B=0.75, K1=1.2)

    # LMIR features
    for term in query:
        wfdoc = doc.count(term)
        idf = np.log(NDOCS / (searcher.reader().doc_frequency(field, term) + 1e-8))
        wfcorp = searcher.reader().frequency(field, term) + 1e-8

        if len(query) > 3:
            LMD = LMD_LONG
        else:
            LMD = LMD_SHORT
        print()
        print("JM", LMD * wfdoc / len(doc) + (1 - LMD) * wfcorp / CORP_LEN)
        print("DIR", (wfdoc + U_DIR * wfcorp / CORP_LEN) / (len(doc) + U_DIR))
        print("KL", -(1 / len(query)) * np.log(wfdoc / len(doc)))
        print()
        docfeats[7] *= LMD * wfdoc / len(doc) + (1 - LMD) * wfcorp / CORP_LEN  # JM
        docfeats[8] *= (wfdoc + U_DIR * wfcorp / CORP_LEN) / (len(doc) + U_DIR)  # DIR
        docfeats[9] += -(1 / len(query)) * np.log(wfdoc / len(doc))  # KL

    return np.log(docfeats)


DOCOFFSET = {}
with open("data/msmarco-docs-lookup.tsv", "rt", encoding="utf8") as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [docid, _, offset] in tsvreader:
        DOCOFFSET[docid] = int(offset)


def getbody(docid, docsfile, index=3):
    docsfile.seek(DOCOFFSET[docid])
    line = docsfile.readline()
    assert line.startswith(docid + "\t"), f"Looking for {docid}, found {line}"
    linelist = line.rstrip().split("\t")
    if len(linelist) == 4:
        return linelist[index]
