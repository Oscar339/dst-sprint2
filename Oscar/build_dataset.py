"""Join the FinBERT shards with the volatility labels into one modelling dataset.

extract_finbert.py saved 114 shard files (250 reports each) on OneDrive. That is
awkward to model with, so this script stacks them into:

  data/processed/dataset.csv        one row per report: key, year, date, cik, name,
                                    logvol_before (baseline input), logvol_after (target),
                                    item7_source, and for each section the number of
                                    chunks kept (n), the true number (true) and where its
                                    rows start in the embedding array (start)
  data/processed/item1_emb.npy      all Item 1 chunk vectors, one row per chunk (float16)
  data/processed/item7_emb.npy      all Item 7 chunk vectors, likewise

Report i's Item 7 chunks are rows  item7_start[i] : item7_start[i] + item7_n[i].
Use load_dataset() below rather than re-doing that bookkeeping.

Usage (from the repo root):  python Oscar/build_dataset.py [--src FOLDER]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DEFAULT_SRC = Path.home() / "OneDrive - University of Bristol" / "DST" / "finbert"
SECTIONS = ["item1", "item7"]


def build(src):
    shards = sorted(src.glob("*_[0-9][0-9][0-9].npz"))
    meta, embs = [], {s: [] for s in SECTIONS}
    for p in shards:
        z = np.load(p)
        part = pd.DataFrame({"key": z["keys"], "item7_source": z["item7_source"]})
        for s in SECTIONS:
            part[f"{s}_n"] = z[f"{s}_n"]
            part[f"{s}_true"] = z[f"{s}_true"]
            embs[s].append(z[f"{s}_emb"])
        meta.append(part)
    meta = pd.concat(meta, ignore_index=True)

    labels = pd.read_csv(PROCESSED / "kogan_edgar_merged.csv",
                         usecols=["key", "date", "name", "cik", "year", "logvol_before", "logvol_after"])
    assert labels["key"].is_unique
    df = meta.merge(labels, on="key", how="left", validate="one_to_one")
    assert df["logvol_after"].notna().all(), "some embedded reports have no label"

    for s in SECTIONS:
        # Where each report's chunks start in the stacked array (a running total).
        df[f"{s}_start"] = np.concatenate([[0], np.cumsum(df[f"{s}_n"])[:-1]])
        arr = np.concatenate(embs[s])
        assert len(arr) == df[f"{s}_n"].sum()
        np.save(PROCESSED / f"{s}_emb.npy", arr)
        print(f"{s}: {len(arr):,} chunks, {arr.nbytes / 1e9:.2f} GB")

    cols = ["key", "year", "date", "cik", "name", "logvol_before", "logvol_after", "item7_source"]
    cols += [f"{s}_{c}" for s in SECTIONS for c in ("n", "true", "start")]
    df[cols].to_csv(PROCESSED / "dataset.csv", index=False)
    print(f"dataset.csv: {len(df):,} reports, {df['year'].min()}-{df['year'].max()}")


def load_dataset(sections=SECTIONS):
    """Return (df, emb) where emb[section] is a memory-mapped (n_chunks, 768) array.

    Memory-mapped means the file stays on disk and only the rows you touch are read,
    so this is quick even though the arrays are over a gigabyte.
    """
    df = pd.read_csv(PROCESSED / "dataset.csv", parse_dates=["date"])
    emb = {s: np.load(PROCESSED / f"{s}_emb.npy", mmap_mode="r") for s in sections}
    return df, emb


def chunks(df, emb, i, section):
    """The chunk vectors of report i (row position in df) for one section."""
    start, n = df[f"{section}_start"].iat[i], df[f"{section}_n"].iat[i]
    return emb[section][start:start + n]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    build(p.parse_args().src)
