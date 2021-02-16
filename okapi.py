import whoosh.scoring


class Okapi(whoosh.scoring.WeightingModel):
    def scorer(self, searcher, fieldname, text, qf):
        return OkapiScorer()

    def final(self, searcher, docnum, score):
        return super().final(searcher, docnum, score)

    def idf(self, searcher, fieldname, text):
        return super().idf(searcher, fieldname, text)


class OkapiScorer(whoosh.scoring.BaseScorer):
    def __init__(self) -> None:
        super().__init__()

    def score(self, matcher):
        # Returns a score for the current document of the matcher.
        for fieldname, termtext in matcher.matching_terms():
            print(termtext)
            return 1 if b"hello" in termtext else 0

    def supports_block_quality(self):
        return False
