import argparse
import csv
import math
import random

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

def display_doc(docid, docsfile):
    docsfile.seek(doc_offset[docid])
    line = docsfile.readline()
    assert line.startswith(docid + "\t"), f"Looking for {docid}, found {line}"
    linelist = line.rstrip().split("\t")
    if len(linelist) == 4:
        print(f"\t{linelist[2]} (@{linelist[1]})")
        print(f"\t{linelist[3][:250]}")


print()
with open("data/msmarco-docs.tsv", encoding="utf-8") as docsfile:
    items = list(queries.items())
    random.shuffle(items)
    # go through the queries and how the model answered them
    for qid, qtext in items:
        if qid in qrels:
            relevant_docs = qrels[qid]
            relevant_docs.sort(key=lambda x: x[1], reverse=True)
            if qid in predictions:
                predicted_docs = predictions[qid]
            else:
                predicted_docs = []
            print("="*50)
            print(f"| Query: '{qtext}'")
            print("="*50)

            predicted_and_irrelevant = [p for p in predicted_docs if p[0] not in [x[0] for x in relevant_docs]]

            print(f"Model found {len(predicted_docs)} documents, "
                  f"of which {len(predicted_docs)-len(predicted_and_irrelevant)} were relevant. "
                  f"There are {len(relevant_docs)} relevant docs in the corpus.")

            if len(predicted_docs) > 0:
                print("-"*50)
                print("| FALSE POSITIVES (highest scoring irrelevant docs)")
                print("-"*50)
                print()
                c = 0
                for docid, rank, score in predicted_and_irrelevant:
                    print(f"Rank {rank} of {len(predicted_docs)}")
                    display_doc(docid, docsfile)
                    c += 1
                    if c >= 5:
                        break

            print()
            print("-"*50)
            print("| FALSE NEGATIVES (lowest scoring relevant docs)")
            print("-"*50)
            print()
            c = 0

            discrepancies = []
            for p in predicted_docs[::-1]:
                for idx, (docid, _) in enumerate(relevant_docs):
                    if p[0] == docid:
                        # rated_too_low = p[1] - idx + 1
                        discrepancies.append((docid, p[1], idx))
                        # if docid not in [x[0] for x in predicted_docs]:
                        #     print(f"Rank {idx+1} of {len(relevant_docs)}")
                        #     display_doc(docid, docsfile)
                        #     c += 1
                        #     if c >= 5:
                        #         break
            # sort by the greatest difference between the judged rank, and the true rank
            discrepancies.sort(key=lambda x: (int(x[1])**.75)-x[2], reverse=True)
            for docid, judged_rank, true_rank in discrepancies[:5]:
                print(f"Judged as {judged_rank} but was {true_rank + 1} of {len(relevant_docs)}")
                display_doc(docid, docsfile)
            input("\nEnter to go to next query...")
            print("\n"*30)
