set -e

model=$1 # c or r (classification or regression)
title=$2 # filename of eval files
if [ "$model" = "r" ]; then 
    python train_svm.py -graded
else
    python train_svm.py
fi
python predict.py sv${model}
./trec_eval -m all_trec data/2019qrels-docs.txt output/sv${model}-predictions.trec > output/sv${model}-trec-eval-${title}.txt
mv output/sv${model}-predictions.trec output/sv${model}-predictions-${title}.trec