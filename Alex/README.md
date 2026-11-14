# Alex

Working notebooks, scratch and dead ends. Not marked directly — this is the
evidence of contribution the brief asks for.

Reading order for my part (scoring and models):

1. `00-Getting-the-data.ipynb` — gets Oscar's `data.pkl` onto your machine
2. `01-Research-and-Plan.md` — the research behind the models, what we decided, and the updated plan
3. `02-Models.ipynb` — the baselines and ridge models, and the comparison with the papers (about 3 minutes)

| File | What it is |
|---|---|
| `scoring.py` | Shared scoring: the training windows, the baselines, the paired bootstrap and the level/spread split. Any model written as `predict(df, train, test)` can use it |
| `ridge_sums.py` | Ridge regression from per-year sums (map/reduce), used by `02-Models.ipynb` |

`02-Models.ipynb` needs `data/processed/data.pkl` (from `00-Getting-the-data.ipynb`)
and downloads Kogan et al.'s small label files into `data/raw/kogan/` the first time.
