"""Ridge regression from per-year sums: fit any training window without touching the reports again.

Why: scikit-learn's RidgeCV decomposes the whole training matrix (up to 21,000 reports x 1,536
numbers) for every fit, about 15 s each. But ridge only needs a few sums over the training reports:

    G   = X^T X      (inputs times inputs)        sx = X^T 1      (input totals)
    sy  = X^T y      (inputs times target)         sb = X^T x_b    (inputs times last year's vol)
    and the scalars n, sum y, sum x_b, sum y^2, sum x_b^2, sum y*x_b.

Sums add up, so they are computed once per filing year (map) and added for any window (reduce).
After that, a fit costs one eigen-decomposition of a p x p matrix (p = 768 or 1,536), whatever the
number of reports.

The target is the straight line's residual r = y - (a + b x_b). Its sums follow from the stored
ones, e.g.  X^T r = sy - a sx - b sb,  so one set of per-year sums serves every window's line.

Inputs are standardised (mean 0, variance 1 on the training rows) as StandardScaler would, and the
penalty is chosen by generalised cross-validation (GCV), the closed-form approximation to
leave-one-out that needs only the eigenvalues (Golub, Heath & Wahba, 1979).
"""
import numpy as np


def year_sums(X, df):
    """Map step: the sums for each filing year. X is (reports x p), rows in df's order."""
    xb, y, years = df["logvol_before"].to_numpy(), df["logvol_after"].to_numpy(), df["year"].to_numpy()
    out = {}
    for yr in np.unique(years):
        m = years == yr
        Xy = X[m].astype(np.float64)
        out[int(yr)] = {
            "n": int(m.sum()), "G": Xy.T @ Xy, "sx": Xy.sum(0), "sy": Xy.T @ y[m], "sb": Xy.T @ xb[m],
            "y": y[m].sum(), "b": xb[m].sum(), "yy": y[m] @ y[m], "bb": xb[m] @ xb[m], "yb": y[m] @ xb[m],
        }
    return out


def subset(sums, cols):
    """The same sums for a subset of the input columns (e.g. Item 1 out of 'both')."""
    out = {}
    for yr, s in sums.items():
        t = dict(s)
        t["G"] = s["G"][np.ix_(cols, cols)]
        for k in ("sx", "sy", "sb"):
            t[k] = s[k][cols]
        out[yr] = t
    return out


def _residual_sums(s, a, b):
    """Sums for the residual r = y - a - b x_b in one year: (X^T r, sum r, sum r^2)."""
    Xr = s["sy"] - a * s["sx"] - b * s["sb"]
    r = s["y"] - a * s["n"] - b * s["b"]
    rr = (s["yy"] + a * a * s["n"] + b * b * s["bb"]
          - 2 * a * s["y"] - 2 * b * s["yb"] + 2 * a * b * s["b"])
    return Xr, r, rr


def fit(sums, years, a, b, alphas, within_year=False):
    """Reduce step + fit: ridge of the line's residual on the standardised inputs.

    within_year=False: centre on the whole window's mean (one intercept), as the first attempt.
    within_year=True:  centre each year on its own mean, so year-to-year differences are removed
                       from both the inputs and the target before fitting.
    Returns a dict with coef (on the raw inputs), mean (the centring vector), alpha, and the GCV
    curve. Predict with  (X - mean) @ coef + offset,  where for within_year `mean` is replaced by
    the test year's own mean.
    """
    p = next(iter(sums.values()))["G"].shape[0]
    G, c, rr, n, sx_tot, r_tot = np.zeros((p, p)), np.zeros(p), 0.0, 0, np.zeros(p), 0.0
    for yr in years:
        s = sums[yr]
        Xr, r, rr_y = _residual_sums(s, a, b)
        if within_year:   # centre this year on its own means
            mu = s["sx"] / s["n"]
            G += s["G"] - s["n"] * np.outer(mu, mu)
            c += Xr - mu * r
            rr += rr_y - r * r / s["n"]
        else:
            G += s["G"]
            c += Xr
            rr += rr_y
            sx_tot += s["sx"]
            r_tot += r
        n += s["n"]
    if within_year:
        mean, offset = None, 0.0
    else:             # centre on the window's means
        mean = sx_tot / n
        G -= n * np.outer(mean, mean)
        c -= mean * r_tot
        rr -= r_tot * r_tot / n
        offset = r_tot / n

    sd = np.sqrt(np.diag(G) / n)                 # population sd, as StandardScaler
    Gs, cs = G / np.outer(sd, sd), c / sd        # sums for the standardised inputs
    eig, V = np.linalg.eigh(Gs)
    eig = np.clip(eig, 0, None)
    q = V.T @ cs
    alphas = np.asarray(alphas, dtype=float)
    shrink = 1.0 / (eig[None, :] + alphas[:, None])               # (alphas x p)
    rss = rr - 2 * (q**2 * shrink).sum(1) + (eig * q**2 * shrink**2).sum(1)
    dof = (eig * shrink).sum(1)                                   # trace of the hat matrix
    gcv = n * rss / (n - dof) ** 2
    k = int(np.argmin(gcv))
    beta_std = V @ (q * shrink[k])
    return {"coef": beta_std / sd, "mean": mean, "offset": offset, "alpha": float(alphas[k]),
            "gcv": gcv, "dof": float(dof[k])}


def predict(model, X_test):
    mean = X_test.mean(axis=0) if model["mean"] is None else model["mean"]
    return (X_test - mean) @ model["coef"] + model["offset"]
