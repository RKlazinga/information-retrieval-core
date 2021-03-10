import csv
from typing import List

import joblib
import numpy as np
import whoosh.scoring
from tqdm import tqdm

import okapi
from singletons import DOCOFFSET, PMI, PREPROCESS, SEARCHER

U_DIR = 2000
LMD_SHORT = 0.65
LMD_LONG = 0.25


def get_query_features(query):
    ndocs = SEARCHER.reader().doc_count_all()

    feats = np.zeros((5, len(query)))

    for i, term in enumerate(query):
        df = SEARCHER.reader().doc_frequency("body", term)
        idf = np.log(ndocs / (df + 1))
        wfcorp = SEARCHER.reader().frequency("body", term) + 1

        feats[0, i] = idf
        feats[1, i] = wfcorp
        feats[2, i] = wfcorp * idf
        feats[3, i] = ndocs / wfcorp

        tpmi = 0
        for term2 in query:
            tpmi += PMI.compute(term, term2)
        feats[4, i] = tpmi

    feats = np.concatenate(
        (np.mean(feats, axis=1), np.max(feats, axis=1), np.sum(feats, axis=1), np.array([np.log(len(query) + 1)]))
    )
    feats = [np.log(score + 1) if score > 0 else -np.log(-score + 1) for score in feats]
    return np.array(feats)


def get_doc_feats(docid, query, FILE, use_title=True):
    body = getbody(docid, FILE)
    doc = [token.text for token in PREPROCESS(body)]
    docfeats = features_per_doc(query, doc)
    if not use_title:
        return docfeats

    title = getbody(docid, FILE, index=2)
    title = [token.text for token in PREPROCESS(title)]
    titlefeats = features_per_doc(query, title, field="title")

    url = getbody(docid, FILE, index=1)
    slashcount = url.count("/")
    notokenlen = len(url)
    url = [token.text for token in PREPROCESS(url.replace(".", " "))]
    urlfeats = features_per_doc(query, url, field="url_text")

    # calculate pointwise mutual info between every word of query and fields
    pmis = []
    for field in [body, title, url]:
        scores = []
        for term in query:
            for word in field:
                scores.append(PMI.compute(term, word))
        if len(scores) > 0:
            pmis += [np.sum(scores), np.max(scores), np.mean(scores)]
        else:
            pmis += [0, 0, 0]
    pmis = [np.log(score + 1) if score > 0 else -np.log(-score + 1) for score in pmis]
    staticfeats = np.array(
        [
            np.log(len(doc) + 1),
            np.log(len(title) + 1),
            np.log(len(url) + 1),
            np.log(slashcount + 1),
            np.log(notokenlen + 1),
            *pmis,
        ]
    )

    return np.concatenate((docfeats, titlefeats, urlfeats, staticfeats))


def features_per_doc(query: List, doc: List, field="body"):
    intersection = set(query) & set(doc)
    if len(intersection) == 0:
        return np.zeros(3 * 8 + 2)

    CORP_LEN = SEARCHER.reader().field_length(field)
    NDOCS = SEARCHER.reader().doc_count_all()
    AVG_DOC_LEN = CORP_LEN / NDOCS
    LMD = LMD_LONG if len(query) > 3 else LMD_SHORT

    docfeats = np.zeros((8, len(intersection)))
    prodfeats = np.ones((2, len(intersection)))

    for i, term in enumerate(intersection):
        wfdoc = doc.count(term)
        df = SEARCHER.reader().doc_frequency(field, term)
        idf = np.log(NDOCS / (df + 1))
        wfcorp = SEARCHER.reader().frequency(field, term) + 1

        # AdaRank features
        docfeats[0, i] = wfdoc
        docfeats[1, i] = idf
        docfeats[2, i] = wfdoc / len(doc) * idf
        docfeats[3, i] = NDOCS / wfcorp
        docfeats[4, i] = wfdoc / len(doc)
        docfeats[5, i] = wfdoc * NDOCS / (len(doc) * wfcorp)
        docfeats[6, i] = okapi.bm25(NDOCS, df, wfcorp, len(doc), AVG_DOC_LEN)

        # LMIR features
        KL = -(1 / len(query)) * np.log(wfdoc / len(doc) + 1)
        if np.isfinite(KL).all():
            docfeats[7, i] = KL

        prodfeats[0, i] = LMD * wfdoc / len(doc) + (1 - LMD) * wfcorp / CORP_LEN  # JM
        prodfeats[1, i] = (wfdoc + U_DIR * wfcorp / CORP_LEN) / (len(doc) + U_DIR)  # DIR

    return np.log(
        np.concatenate(
            (np.mean(docfeats, axis=1), np.max(docfeats, axis=1), np.sum(docfeats, axis=1), np.prod(prodfeats, axis=1))
        )
        + 1
    )


def getbody(docid, docsfile, index=3):
    docsfile.seek(DOCOFFSET[docid])
    line = docsfile.readline()
    assert line.startswith(docid + "\t"), f"Looking for {docid}, found {line}"
    linelist = line.rstrip().split("\t")
    if len(linelist) == 4:
        return linelist[index]
