set -e

model=$1 # svc, svr, or clusvm
title=$2 # filename of eval files
if [ "$model" = "svr" ]; then 
    python train_svm.py -graded
elif [ "$model" = "clusvm" ]; then 
    python train_cluster_svm.py -graded
else
    python train_svm.py
fi
python predict.py ${model}
./trec_eval -m all_trec data/2019qrels-docs.txt output/${model}-predictions.trec > output/${model}-trec-eval-${title}.txt
mv output/${model}-predictions.trec output/${model}-predictions-${title}.trec