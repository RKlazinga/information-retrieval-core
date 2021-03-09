import gc
import gzip
import pickle
import pickletools
import time


def save(filepath, obj):
    gc.disable()
    with gzip.open(filepath, "wb") as f:
        pickled = pickle.dumps(obj)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)
    gc.enable()


def load(filepath):
    print("loading", filepath, "...")
    tik = time.time()
    gc.disable()
    with gzip.open(filepath, "rb") as f:
        p = pickle.Unpickler(f)
        obj = p.load()
    gc.enable()
    print("took:", time.time() - tik)
    return obj


CHUNK_SIZE = 1_000_000


def chunks(iterable, n=CHUNK_SIZE):
    args = [iter(iterable)] * int(n)
    return zip(*args)


if __name__ == "__main__":
    import glob

    import joblib

    from pmi import PMI

    for filepath in glob.glob("pmi*.pkl"):
        something = joblib.load(filepath)
        save(filepath.replace(".pkl", "-new.pkl"), something)

