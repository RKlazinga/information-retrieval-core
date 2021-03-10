import csv
import subprocess

query_path = "data/msmarco-test2019-queries.tsv"


qrels = "test-qrels.txt"
model1 = "output/svm-predictions-reranking.trec"
model2 = "output/svc-predictions-pmi-c.trec"

m1_results = {}
m2_results = {}

proc1 = subprocess.Popen(["trec_eval", "-m", "all_trec", "-q", qrels, model1], stdout=subprocess.PIPE)
for line in proc1.stdout:
    metric, qid, score = [x.strip(" \t\r\n") for x in line.decode("utf-8").split("\t")]
    if metric == "ndcg_cut_10":
        if qid != "all":
            m1_results[qid] = float(score)

proc2 = subprocess.Popen(["trec_eval", "-m", "all_trec", "-q", qrels, model2], stdout=subprocess.PIPE)
for line in proc2.stdout:
    metric, qid, score = [x.strip(" \t\r\n") for x in line.decode("utf-8").split("\t")]
    if metric == "ndcg_cut_10":
        if qid != "all":
            m2_results[qid] = float(score)

counter = 0
total = 0
for q in m1_results.keys():
    if m2_results[q] > m1_results[q]:
        counter += 1
    total += 1

print(f"Model 2 outperformed model 1 in {counter} out of {total} queries")