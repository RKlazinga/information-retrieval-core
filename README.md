# IN4325 Information Retrieval

Authors:
Rembrandt Klazinga (@RKlazinga)
Hans Brouwer (@JCBrouwer)

### To reproduce results presented in our report:

```language=bash

# download the MSMARCO 2019 dataset (this will take a while)
mkdir data
cd data
wget https://trec.nist.gov/data/deep/2019qrels-docs.txt &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctest2019-top100.gz &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
gunzip *.gz
cd ..

# install dependencies
pip install numpy sklearn whoosh tqdm

# construct dataset index (this will also take a while)
python index.py data/masmarco-docs.tsv
# python index.py data/masmarco-docs.tsv -num_docs 100000 # quicker test index

# run predictions with proabilistic models (our bm25 implementation and whoosh's bm25)
python predict.py okapi
python predict.py bm25

# train and run adarank (our implementation of adarank seems to be broken)
python train_adarank.py
python predict.py adarank

# generate dataset, train, and predict with SVM model
python triples.py 10000
python train_svm.py
python predict.py svm

# evaluate outputs for all models
git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval
make
cd ..
for model in okapi bm25 svm adarank; do
./trec_eval/trec_eval data/2019qrels-docs.txt output/${model}-predictions.trec > output/${model}-trec-eval.txt
done
```

## Acknowledgments

Mutliple features are based on the implementations found in this repo: https://github.com/jax79sg/IRDM2017

The (broken) AdaRank implementation is based on this repo: https://github.com/rueycheng/AdaRank.git