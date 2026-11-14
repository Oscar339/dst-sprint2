"""Shared scoring for every Sprint 2 model, so all models are judged the same way.

A model is a function   predict(df, train, test) -> predicted logvol_after for the test rows,
where `train` and `test` are boolean masks over the reports (rows of data.pkl). `evaluate` runs it
on every test year and training window, and compares it with a baseline using a paired bootstrap.

Usage (from a notebook in Alex/, Louis/ or report/), with Oscar's data.pkl:
    import pickle, sys; sys.path.insert(0, "../Alex")
    import scoring
    with open("../data/processed/data.pkl", "rb") as f:
        d = pickle.load(f)
    df = scoring.frame(d)          # year, logvol_before, logvol_after, key, name, cik, date
    results = scoring.evaluate(df, my_predict, name="my model")
"""
import numpy as np
import pandas as pd


def frame(d):
    """One row per report from data.pkl, with the column names the helpers here use."""
    return pd.concat([pd.DataFrame({"year": d["year"], "logvol_before": d["baseline"], "logvol_after": d["y"]}),
                      d["info"].reset_index(drop=True)], axis=1)


FIRST_YEAR = 1996
TEST_YEARS = range(2001, 2007)

# Training windows. Each maps a test year T to the filing years used for training.
#   paper / strict      : the two windows agreed in report/02-Data.ipynb (data/output/splits.csv)
#   last year, all prior: one year vs all the history, both with strict's one-year gap, so no
#                         training label (12 months after filing) overlaps the test year
WINDOWS = {
    "paper": lambda T: range(T - 5, T),
    "strict": lambda T: range(T - 5, T - 1),
    "last year": lambda T: range(T - 2, T - 1),
    "all prior": lambda T: range(FIRST_YEAR, T - 1),
}


def masks(df, test_year, window):
    """Boolean masks (train, test) for one test year and window."""
    years = WINDOWS[window](test_year)
    return df["year"].isin(years).to_numpy(), (df["year"] == test_year).to_numpy()


def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def bootstrap_diff(y, p_model, p_base, n_boot=2000, seed=0):
    """Paired bootstrap of MSE(model) - MSE(baseline) over the test reports.

    The same resampled reports are used for both models, so report-to-report noise cancels and
    the interval is much narrower than comparing two separate intervals. Returns
    (difference, 2.5% point, 97.5% point). Negative means the model has lower error.
    """
    d = (np.asarray(y) - np.asarray(p_model)) ** 2 - (np.asarray(y) - np.asarray(p_base)) ** 2
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


# ---------------------------------------------------------------- baselines

def predict_copy(df, train, test):
    """Next year's volatility = last year's. Needs no training."""
    return df["logvol_before"].to_numpy()[test]


def fit_line(df, train):
    """Least-squares line logvol_after = a + b * logvol_before on the training rows. Returns (a, b)."""
    x, y = df["logvol_before"].to_numpy()[train], df["logvol_after"].to_numpy()[train]
    b, a = np.polyfit(x, y, 1)
    return a, b


def predict_line(df, train, test):
    a, b = fit_line(df, train)
    return a + b * df["logvol_before"].to_numpy()[test]


# ---------------------------------------------------------------- running a model

def evaluate(df, predict, name, baseline=predict_line, windows=None, test_years=TEST_YEARS,
             n_boot=2000):
    """Score `predict` on every (test year, window), against `baseline`.

    `predict` and `baseline` are called as f(df, train, test). Returns one row per
    (test year, window) with both MSEs, the difference, its 95% bootstrap interval and the
    % change in MSE (negative = the model is better than the baseline).

    Each MSE also splits exactly into two parts, MSE = level + spread:
      level  = (mean error)^2, how far off the whole test year's average volatility is. It is
               one number per year, shared by every company, so no single report can know it;
      spread = variance of the errors, how well the model ranks companies within the year.
    `spread_change_pct` (with its own paired-bootstrap interval) compares the spread parts only.
    """
    y_all = df["logvol_after"].to_numpy()
    rows = []
    for window in windows or WINDOWS:
        for T in test_years:
            train, test = masks(df, T, window)
            y = y_all[test]
            p, p0 = predict(df, train, test), baseline(df, train, test)
            e, e0 = y - p, y - p0
            diff, lo, hi = bootstrap_diff(y, p, p0, n_boot=n_boot)
            # Shift each model's predictions by its own mean error: what is left is the spread part.
            s_diff, s_lo, s_hi = bootstrap_diff(y, p + e.mean(), p0 + e0.mean(), n_boot=n_boot)
            rows.append({"model": name, "window": window, "test_year": T,
                         "train_reports": int(train.sum()), "test_reports": int(test.sum()),
                         "mse": mse(y, p), "baseline_mse": mse(y, p0),
                         "diff": diff, "ci_low": lo, "ci_high": hi,
                         "change_pct": 100 * diff / mse(y, p0),
                         "level": float(e.mean() ** 2), "baseline_level": float(e0.mean() ** 2),
                         "spread": float(e.var()), "baseline_spread": float(e0.var()),
                         "spread_change_pct": 100 * s_diff / float(e0.var()),
                         "spread_ci_low_pct": 100 * s_lo / float(e0.var()),
                         "spread_ci_high_pct": 100 * s_hi / float(e0.var())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- shared input helper

def mean_pool(df, emb, section):
    """Average each report's chunk vectors: a (reports x 768) float32 table."""
    starts = df[f"{section}_start"].to_numpy()
    sums = np.add.reduceat(emb[section], starts, axis=0, dtype=np.float32)
    return sums / df[f"{section}_n"].to_numpy()[:, None]
