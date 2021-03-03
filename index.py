import argparse
import os.path

from tqdm import tqdm
from whoosh import index, writing
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, TEXT, Schema
from whoosh.filedb.filestore import FileStorage
from whoosh.writing import AsyncWriter

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
parser.add_argument("-num_docs", type=int, default=None)
parser.add_argument("-threads", type=int, default=1)  # seems using more than 1 thread may be broken?
parser.add_argument("-reload", action="store_true")
args = parser.parse_args()

schema = Schema(
    docid=ID(stored=True),
    url=ID(stored=True),
    title=TEXT(stored=True, analyzer=StemmingAnalyzer()),  # maybe no stemming here?
    body=TEXT(analyzer=StemmingAnalyzer()),
)

index_dir = "data/msmarcoidx" if args.num_docs is None else "data/quickidx"
if not os.path.exists(index_dir):
    os.mkdir(index_dir)
    index.create_in(index_dir, schema)
    args.reload = True

storage = FileStorage(index_dir)
# Open an existing index
ix = storage.open_index()

ix.writer().commit(mergetype=writing.CLEAR)

print(f"Loading documents from {args.data}")
writers = [AsyncWriter(ix) for _ in range(args.threads)]
with open(args.data, "r", encoding="utf-8") as docs:
    i = 0
    line = docs.readline()
    pbar = tqdm(total=args.num_docs if args.num_docs is not None else 3_213_835)
    while line != "" and (args.num_docs is None or i < args.num_docs):
        docid, url, title, body = line.split("\t")
        writers[i % args.threads].add_document(docid=docid, url=url, title=title, body=body)
        line = docs.readline()
        i += 1
        pbar.update(1)
pbar.set_description("Committing...")
pbar.refresh()

[w.commit() for w in writers]
