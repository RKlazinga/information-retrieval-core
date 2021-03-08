import csv
from typing import List

import joblib
import numpy as np
import whoosh.scoring
from tqdm import tqdm

import okapi

pmi = None

U_DIR = 2000
LMD_SHORT = 0.65
LMD_LONG = 0.25

preprocess = whoosh.analysis.StemmingAnalyzer()


def get_doc_feats(docid, query, FILE, SEARCHER, use_title=True):
    global pmi
    if pmi is None:
        pmi = joblib.load("pmi.pkl")

    body = getbody(docid, FILE)
    doc = [token.text for token in preprocess(body)]
    docfeats = features_per_doc(query, doc, SEARCHER)
    if not use_title:
        return docfeats

    title = getbody(docid, FILE, index=2)
    title = [token.text for token in preprocess(title)]
    titlefeats = features_per_doc(query, title, SEARCHER, field="title")

    url = getbody(docid, FILE, index=1)
    slashcount = url.count("/")
    notokenlen = len(url)
    url = [token.text for token in preprocess(url.replace(".", " "))]
    urlfeats = features_per_doc(query, url, SEARCHER, field="url_text")

    # calculate pointwise mutual info between every word of query and fields
    pmis = []
    for field in [body, title, url]:
        max_pmi = -20
        for term in query:
            for word in field:
                score = pmi.compute(term, word)
                if score > max_pmi:
                    max_pmi = score
        pmis.append(max_pmi)
    staticfeats = np.array(
        [
            np.log(len(doc) + 1e-8),
            np.log(len(title) + 1e-8),
            np.log(len(url) + 1e-8),
            np.log(slashcount + 1e-8),
            np.log(notokenlen + 1e-8),
            *pmis,
        ]
    )

    return np.concatenate((docfeats, titlefeats, urlfeats, staticfeats))


# def get_url_corpus_length(searcher):
#     url_corp_len = 0
#     for fields in tqdm(searcher.documents()):
#         url = [token.text for token in preprocess(fields["url"])]
#         url_corp_len += len(url)
#     print(url_corp_len)
# URL_CORP_LEN = 21_177_607


def features_per_doc(query: List, doc: List, searcher, field="body"):
    CORP_LEN = searcher.reader().field_length(field)
    NDOCS = searcher.reader().doc_count_all()
    AVG_DOC_LEN = CORP_LEN / NDOCS

    intersection = set(query) & set(doc)
    docfeats = np.ones(10)

    for term in intersection:
        wfdoc = doc.count(term)
        df = searcher.reader().doc_frequency(field, term)
        idf = np.log(NDOCS / (df + 1e-8))
        wfcorp = searcher.reader().frequency(field, term) + 1e-8

        # AdaRank features
        docfeats[0] += wfdoc
        docfeats[1] += idf
        docfeats[2] += wfdoc / len(doc) * idf
        docfeats[3] += NDOCS / wfcorp
        docfeats[4] += wfdoc / len(doc)
        docfeats[5] += wfdoc * NDOCS / (len(doc) * wfcorp)
        docfeats[6] += okapi.bm25(NDOCS, df, wfcorp, len(doc), AVG_DOC_LEN)

        # LMIR features
        LMD = LMD_LONG if len(query) > 3 else LMD_SHORT
        docfeats[7] *= LMD * wfdoc / len(doc) + (1 - LMD) * wfcorp / CORP_LEN  # JM
        docfeats[8] *= (wfdoc + U_DIR * wfcorp / CORP_LEN) / (len(doc) + U_DIR)  # DIR
        KL = -(1 / len(query)) * np.log(wfdoc / len(doc) + 1e-8)
        if np.isfinite(KL).all():
            docfeats[9] += KL

    return np.log(docfeats + 1e-8)


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
