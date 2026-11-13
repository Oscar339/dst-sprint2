"""Numbers quoted in report/01-Introduction.ipynb. Run from repo root: python Oscar/intro_numbers.py"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

P = Path("data/processed")
FINBERT = Path.home() / "OneDrive - University of Bristol" / "DST" / "finbert"

df = pd.read_csv(P / "dataset.csv")
merged = pd.read_csv(P / "kogan_edgar_merged.csv").set_index("key")
emb = np.load(P / "item7_emb.npy", mmap_mode="r")

# Item 7 words come from Kogan for reports whose EDGAR Item 7 was replaced
words7 = np.where(df["item7_source"] == "kogan",
                  merged.loc[df["key"], "kogan_item7_words"], merged.loc[df["key"], "edgar_item7_words"])
words = merged.loc[df["key"], "edgar_item1_words"].to_numpy() + words7

log = (FINBERT / "run_main.log").read_text()
stamps = re.findall(r"^=== .* at (.+)$|^=== launch (.+)$", log, re.M)  # "=== launch <time>" ... "=== ... at <time>"
stamps = [datetime.strptime(a or b, "%a %b %d %H:%M:%S GMT %Y") for a, b in stamps]
start, end = stamps[0], stamps[-1]

print(f"reports:                {len(df):,}")
print(f"years:                  {df['year'].min()}-{df['year'].max()}")
print(f"median words (1 + 7):   {np.median(words):,.0f}")
print(f"chunk size (tokens):    {510}")
print(f"median chunks (1 + 7):  {(df['item1_true'] + df['item7_true']).median():.0f}")
print(f"vector length:          {emb.shape[1]}")
print(f"FinBERT run time:       {(end - start).total_seconds() / 3600:.1f} h")
