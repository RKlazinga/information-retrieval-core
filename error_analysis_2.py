import argparse
import csv
import subprocess
from whoosh.analysis import StemmingAnalyzer


parser = argparse.ArgumentParser()
parser.add_argument("-qrel_file", type=str)
parser.add_argument("-prediction_file", type=str)
args = parser.parse_args()

query_path = "data/msmarco-test2019-queries.tsv"

print("Reading in files...")
# read in all queries
queries = dict()
with open(query_path) as query_file:
    for line in query_file.read(-1).split("\n"):
        if line != "":
            key, value = line.split("\t")
            queries[key] = value

# read in all qrels for the test set
qrels = dict()
with open(args.qrel_file) as qrel_file:
    for line in qrel_file.read(-1).split("\n"):
        if line != "":
            qid, _, docid, relevance = line.split(" ")
            relevance = int(relevance)
            if relevance > 0:
                if qid not in qrels:
                    qrels[qid] = [(docid, relevance)]
                else:
                    qrels[qid].append((docid, relevance))

# read in the predictions
predictions = dict()
with open(args.prediction_file) as prediction_file:
    for line in prediction_file.read(-1).split("\n"):
        if line != "":
            qid, _, docid, rank, score, _ = line.split(" ")
            score = float(score)
            if qid not in predictions:
                predictions[qid] = [(docid, rank, score)]
            else:
                predictions[qid].append((docid, rank, score))

# read in offset file
doc_offset = {}
with open("data/msmarco-docs-lookup.tsv", "rt", encoding="utf8") as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [docid, _, offset] in tsvreader:
        doc_offset[docid] = int(offset)

proc = subprocess.Popen(["trec_eval", "-m", "all_trec", "-q", args.qrel_file, args.prediction_file], stdout=subprocess.PIPE)
metrics = ["map", "ndcg_cut_10", "recip_rank"]
results = {
    m: {} for m in metrics
}
average = {}
for line in proc.stdout:
    metric, qid, score = [x.strip(" \t\r\n") for x in line.decode("utf-8").split("\t")]
    if metric in metrics:
        if qid == "all":
            average[metric] = float(score)
        else:
            results[metric][qid] = float(score)

qids = list(results[metrics[0]].keys())
# sort qids by their word map and ndcg score (normalised by the average)
qids.sort(key=lambda q: results["map"][q]/average["map"] + results["ndcg_cut_10"][q]/average["ndcg_cut_10"])


def display_doc(docid, docsfile):
    docsfile.seek(doc_offset[docid])
    line = docsfile.readline()
    assert line.startswith(docid + "\t"), f"Looking for {docid}, found {line}"
    linelist = line.rstrip().split("\t")
    if len(linelist) == 4:
        print(f"\t{linelist[2]} (@{linelist[1]})")
        print(f"\t{linelist[3][:250]}")


preprocess = StemmingAnalyzer()
with open("data/msmarco-docs.tsv", encoding="utf-8") as docsfile:
    for qid in qids:
        qtext = queries[qid]
        relevant_docs = qrels[qid]
        relevant_docs.sort(key=lambda x: x[1], reverse=True)
        if qid in predictions:
            predicted_docs = predictions[qid]
        else:
            predicted_docs = []

        show_per_category = 5

        print("=" * 50)
        print(f"| Query:       '{qtext}'")
        print(f"| Tokenized:   {[w.text for w in preprocess(qtext)]}")
        print(f"| Performance: {' '.join([m + '=' + str(results[m][qid]) for m in metrics])}")
        print("=" * 50)

        print("-" * 50)
        print("| HIGHLY RANKED AND RELEVANT")
        print("-" * 50)
        c = 0
        for docid, rank, score in predicted_docs:
            if docid in [x[0] for x in relevant_docs]:
                print(f"Ranked {rank} of {len(predicted_docs)}")
                display_doc(docid, docsfile)
                c += 1
                if c >= show_per_category:
                    break

        if len(predicted_docs) > 0:
            print("-" * 50)
            print("| HIGHLY RANKED NON-RELEVANT")
            print("-" * 50)
            print()
            c = 0
            for docid, rank, score in predicted_docs:
                if docid not in [x[0] for x in relevant_docs]:
                    print(f"Rank {rank} of {len(predicted_docs)}")
                    display_doc(docid, docsfile)
                    c += 1
                    if c >= show_per_category:
                        break

        print()
        print("-" * 50)
        print("| LOW RANK, HIGHLY RELEVANT")
        print("-" * 50)
        print()
        c = 0

        discrepancies = []
        for p in predicted_docs[::-1]:
            for idx, (docid, _) in enumerate(relevant_docs):
                if p[0] == docid:
                    discrepancies.append((docid, p[1], idx))
        # sort by the greatest difference between the judged rank, and the true rank
        discrepancies.sort(key=lambda x: (int(x[1])) - x[2], reverse=True)
        for docid, judged_rank, true_rank in discrepancies[:show_per_category]:
            if int(judged_rank) > true_rank:
                print(f"Judged as {judged_rank} but was {true_rank + 1} of {len(relevant_docs)}")
                display_doc(docid, docsfile)
        input("\nEnter to go to next query...")
        print("\n" * 30)