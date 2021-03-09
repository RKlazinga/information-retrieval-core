import numpy as np
import whoosh.scoring


def bm25(N, df, tf, l_d, l_avg, k1=1.2, b=0.75):
    """https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html

    Args:
        N (int): number of documents
        df (int): frequency of term in all documents
        tf (int): frequency of term in this document
        l_d (int): length of this document
        l_avg (int): length of the average document
        k1 (float): hyperparam: impact of document frequency
        b (float): hyperparam: determines scaling of document length, [0, 1]

    Returns:
        Okapi BM25 score
    """
    return np.log(N / df) * ((k1 + 1) * tf) / (k1 * ((1 - b) + b * l_d / l_avg) + tf)


class OkapiWeighting(whoosh.scoring.WeightingModel):
    def scorer(self, searcher, fieldname, text, qf=1):
        return self.OkapiScorer(searcher, fieldname, text, qf=qf)

    class OkapiScorer(whoosh.scoring.BaseScorer):
        def __init__(self, searcher, fieldname, text, qf=1):
            self.searcher = searcher
            self.fieldname = fieldname
            self.text = text
            self.qf = qf

        def score(self, matcher):
            docid = matcher.id()

            numdocs = self.searcher.doc_count_all()
            length = self.searcher.doc_field_length(docid, self.fieldname)
            avg_length = self.searcher.avg_field_length(self.fieldname)

            df = self.searcher.reader().term_info(self.fieldname, self.text).doc_frequency()
            freq_in_doc = matcher.weight()

            return bm25(numdocs, df, freq_in_doc, length, avg_length)

        def max_quality(self):
            return 10000


weighting = OkapiWeighting()
