"""Pack everything a model needs into ONE file: data/processed/data.pkl

Builds on the output of build_dataset.py (dataset.csv + item1_emb.npy + item7_emb.npy),
so run that first. Usage (from the repo root):  python Oscar/make_handover.py

Load it with:
    import pickle
    with open("data/processed/data.pkl", "rb") as f:
        d = pickle.load(f)

What's inside (one entry per report, all in the same order, 27,142 reports):
    d["y"]         logvol_after: volatility in the 12 months after the report (what we predict)
    d["baseline"]  logvol_before: volatility in the 12 months before (the number to beat)
    d["year"]      filing year, for train/test splits (train on earlier years, test on later)
    d["X"]         (reports x 1536) each section's chunks averaged: Item 1 (cols 0-767),
                   then Item 7 (cols 768-1535). Ready for scikit-learn or a simple MLP.
    d["X_item1"]   list: d["X_item1"][i] is report i's Item 1 chunks, shape (n_chunks, 768).
    d["X_item7"]   list: the same for Item 7. These two are for the attention network.
    d["info"]      table of key, name, cik, date, so you can see which company a row is.
"""
import pickle

import numpy as np

from build_dataset import PROCESSED, load_dataset


def split_chunks(df, arr, section):
    """Cut the stacked chunk table into one array per report."""
    starts, ns = df[f"{section}_start"].to_numpy(), df[f"{section}_n"].to_numpy()
    return [np.array(arr[s:s + n]) for s, n in zip(starts, ns)]   # np.array copies off the disk-backed file


df, emb = load_dataset()
chunks = {s: split_chunks(df, emb[s], s) for s in ["item1", "item7"]}

# Average each report's chunks into one vector per section, then put the two sections side by side.
# (Averaging in float32, because summing many float16 numbers loses precision.)
mean = {s: np.stack([c.astype(np.float32).mean(axis=0) for c in chunks[s]]) for s in chunks}
X = np.hstack([mean["item1"], mean["item7"]])

d = {
    "y": df["logvol_after"].to_numpy(),
    "baseline": df["logvol_before"].to_numpy(),
    "year": df["year"].to_numpy(),
    "X": X,
    "X_item1": chunks["item1"],
    "X_item7": chunks["item7"],
    "info": df[["key", "name", "cik", "date"]],
}

out = PROCESSED / "data.pkl"
with open(out, "wb") as f:
    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"wrote {out}  ({out.stat().st_size / 1e9:.2f} GB)")
print(f"X {X.shape}, y {d['y'].shape}, {len(d['X_item7'])} chunk lists per section")
