"""Source the numbers for the realistic scaling ladder in report/04-Scaling.ipynb, section 6.

Two questions the scaling argument needs answered with data, not guesses:

  1. How many 10-Ks have been filed each year, up to today?  -> the SEC's own EDGAR index.
     The quarterly index lists every filing by form type; we stream it (about 700 MB compressed in total, not stored)
     and count the annual-report forms. 10-K405 (until 2002) was a variant of the 10-K, so it is counted too,
     separately, so you can see the switch-over. Amendments (/A) are not counted. Note: the small-company 10-KSB
     (until 2009) is filed in the index as "10KSB", without the hyphen, so the "10-KSB" names below match nothing
     and those reports are NOT counted. That is deliberate in effect: neither Kogan's corpus nor EDGAR-CORPUS has
     them, and our 1997-2006 counts match EDGAR-CORPUS's to within about 2%.

  2. How long are reports now, and how much of a 10-K are Items 1 and 7?  -> EDGAR-CORPUS (Loukas et al., 2021).
     We stream the 'test' split (a random sample, about a tenth of each year) for an old, a middle and a recent year,
     and count the words in every section.

Saves two small tables to data/output/ (committed): scaling-sec-10k-counts.csv and scaling-edgar-lengths.csv.
Takes about 10-20 minutes, almost all of it download.

Usage (from the repo root, with the shared venv):  python Oscar/scale_sources.py
"""
import gzip
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "output"

# The SEC asks automated clients to identify themselves; a generic name, no personal details.
HEADERS = {"User-Agent": "DST coursework scaling check admin@example.org"}
SEC_INDEX = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/form.gz"
ANNUAL_FORMS = ["10-K", "10-K405", "10-KSB", "10-KSB40", "10-KT", "10-KT405"]
SEC_YEARS = range(1996, 2026)

EDGAR_URL = "https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/{year}/test.jsonl"
EDGAR_YEARS = [2006, 2013, 2020]


def sec_counts():
    rows = []
    for year in SEC_YEARS:
        for q in range(1, 5):
            req = urllib.request.Request(SEC_INDEX.format(year=year, q=q), headers=HEADERS)
            counts = dict.fromkeys(ANNUAL_FORMS, 0)
            with urllib.request.urlopen(req) as r:
                for line in io.TextIOWrapper(gzip.GzipFile(fileobj=r), encoding="latin-1"):
                    form = line[:12].strip()          # the form type is the first column
                    if form in counts:
                        counts[form] += 1
            rows += [{"year": year, "quarter": q, "form": f, "filings": n} for f, n in counts.items()]
            time.sleep(0.2)                           # well under the SEC's 10 requests a second
        y = sum(r["filings"] for r in rows if r["year"] == year)
        print(f"SEC {year}: {y:,} annual reports (10-K and variants)")
    out = pd.DataFrame(rows).groupby(["year", "form"], as_index=False)["filings"].sum()
    out.to_csv(OUTPUT / "scaling-sec-10k-counts.csv", index=False)


def edgar_lengths():
    rows = []
    for year in EDGAR_YEARS:
        with urllib.request.urlopen(EDGAR_URL.format(year=year)) as r:
            for line in r:
                rec = json.loads(line)
                sections = {k: v for k, v in rec.items() if k.startswith("section_")}
                words = {k: len(v.split()) if v else 0 for k, v in sections.items()}
                rows.append({"year": year,
                             "item1": words.get("section_1", 0),
                             "item1A": words.get("section_1A", 0),
                             "item7": words.get("section_7", 0),
                             "all_sections": sum(words.values())})
        print(f"EDGAR-CORPUS {year}: {sum(r['year'] == year for r in rows):,} reports in the test split")
    df = pd.DataFrame(rows)
    df["items_1_7"] = df["item1"] + df["item7"]
    df = df[df["all_sections"] >= 1000]               # drop near-empty extractions (cover pages, failed parses)
    summary = df.groupby("year").agg(
        reports_sampled=("all_sections", "size"),
        median_item1=("item1", "median"), median_item1A=("item1A", "median"), median_item7=("item7", "median"),
        median_items_1_7=("items_1_7", "median"), median_all_sections=("all_sections", "median"),
        mean_items_1_7=("items_1_7", "mean"), mean_all_sections=("all_sections", "mean"),
        share_with_item1A=("item1A", lambda v: float(np.mean(v >= 100))))
    summary.to_csv(OUTPUT / "scaling-edgar-lengths.csv")
    print(summary.round(0).to_string())


if __name__ == "__main__":
    edgar_lengths()
    sec_counts()
