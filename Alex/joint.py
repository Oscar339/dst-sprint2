"""Fitting last year's volatility and the text together, within each year.

The two-step models of 02 and 03 fit the straight line y = a + b x first (x = logvol_before) and then fit the
text to the line's residual. When the text and x are correlated, that is not the same as fitting both together:
by the Frisch-Waugh-Lovell theorem the joint coefficient on the text comes from removing x from y *and* from the
text, and the two-step fit removes it from y only, so the text can never correct b. This module has the joint
versions:

    make_ridge(df, X, kind)            kind = "two-step" (02's within-year ridge), "joint" or "no text"
    make_attention(df, chunks, seed)   the attention network of 03 with its own unpenalised slope on x

Both work within year (each year's own means removed), so the text only changes how companies are ranked; the
prediction is the straight line's average for the test year plus the within-year part, so the level is exactly
the line's. lambda (ridge) or the epoch count (network) is chosen on the last training year, then the model is
refitted on the whole window, as in 02. make_ridge(..., pooled=True) is the exception: all training years are
fitted together, as 02's pooled ridge, so the model predicts each test year's level as well.
"""
import copy
import time

import numpy as np
import pandas as pd

import scoring

ALPHAS = np.logspace(0, 9, 37)                 # the grid of 02


def centre(v, years):
    """Subtract each year's own mean (rows of v grouped by years)."""
    v = np.asarray(v, dtype=np.float64)
    out = v.copy()
    for yr in np.unique(years):
        m = years == yr
        out[m] = v[m] - v[m].mean(axis=0)
    return out


# ---------------------------------------------------------------- ridge

def ridge_path(Z, t, alphas):
    """Ridge coefficients of t on Z for every alpha (no intercept: everything is centred)."""
    eig, V = np.linalg.eigh(Z.T @ Z)
    q = V.T @ (Z.T @ t)
    return V @ (q[:, None] / (np.clip(eig, 0, None)[:, None] + np.asarray(alphas)[None, :]))   # p x len(alphas)


def fit_ridge(kind, X, x, y, years, alphas, b=None, pooled=False):
    """One fit on these rows. kind: 'two-step', 'joint' or 'no text'.
    Within year (default), each year's own means are removed; pooled, the window's overall means are, so the
    intercept is unpenalised and the model also predicts the year's level.
    Returns sd (text standardisation), beta (text, p x len(alphas)) and gamma (slope on x, per alpha); pooled
    fits also keep the training means, which the prediction needs."""
    if pooled:                                 # one group: centre on the whole window's means
        years = np.zeros(len(x), dtype=int)
    xc, yc = centre(x, years), centre(y, years)
    mu = {"pooled": pooled, "y": float(np.mean(y)), "x": float(np.mean(x)),
          "X": np.asarray(X, dtype=np.float64).mean(axis=0) if (pooled and X is not None) else None}
    if kind == "no text":
        return {"sd": None, "beta": None, "gamma": np.full(len(alphas), (xc @ yc) / (xc @ xc)), "mu": mu}
    Xc = centre(X, years)
    sd = np.sqrt((Xc ** 2).mean(axis=0)) + 1e-12   # population sd of the within-year inputs, as ridge_sums.py
    Z = Xc / sd
    if kind == "two-step":                     # the text explains the line's residual; slope fixed at b
        return {"sd": sd, "beta": ridge_path(Z, yc - b * xc, alphas), "gamma": np.full(len(alphas), b), "mu": mu}
    if kind != "joint":
        raise ValueError(kind)
    # joint, x unpenalised: remove x from y and from every column of Z (Frisch-Waugh-Lovell), ridge on the rest
    k = xc @ xc
    Zt, yt = Z - np.outer(xc, (xc @ Z) / k), yc - xc * (xc @ yc) / k
    beta = ridge_path(Zt, yt, alphas)
    return {"sd": sd, "beta": beta, "gamma": (xc @ yc - (xc @ Z) @ beta) / k, "mu": mu}


def within_part(model, X, x, years, j=0):
    """The within-year prediction (centred on each row's own year) for alpha index j."""
    p = model["gamma"][j] * centre(x, years)
    if model["beta"] is not None:
        p = p + (centre(X, years) / model["sd"]) @ model["beta"][:, j]
    return p


def pooled_prediction(model, X, x, j=0):
    """A pooled fit's full prediction (level included): the training means plus the fitted deviations."""
    mu = model["mu"]
    p = mu["y"] + model["gamma"][j] * (x - mu["x"])
    if model["beta"] is not None:
        p = p + ((X - mu["X"]) / model["sd"]) @ model["beta"][:, j]
    return p


def make_ridge(df, X, kind, alphas=ALPHAS, log=None, tag="", pooled=False):
    """A predict(df, train, test) for scoring.evaluate. pooled=True fits all training years together (no
    within-year centring), so the model also predicts the test year's level; lambda is then chosen on the
    last training year's raw MSE rather than its within-year MSE."""
    x_all, y_all, yr_all = df["logvol_before"].to_numpy(), df["logvol_after"].to_numpy(), df["year"].to_numpy()

    def predict(df_, train, test):
        years = sorted(set(yr_all[train]))
        sub, val = train & (yr_all != years[-1]), train & (yr_all == years[-1])
        rows = lambda m: (X[m] if X is not None else None, x_all[m], y_all[m], yr_all[m])
        a, b = scoring.fit_line(df_, train)
        _, b_s = scoring.fit_line(df_, sub)    # lambda: fit on all but the last year (with their own line), score it
        grid = alphas if kind != "no text" else alphas[:1]
        m_s = fit_ridge(kind, *rows(sub), grid, b_s, pooled=pooled)
        Xv, xv, yv, yrv = rows(val)
        if pooled:
            curve = [np.mean((yv - pooled_prediction(m_s, Xv, xv, j)) ** 2) for j in range(len(grid))]
        else:
            yv_c = centre(yv, yrv)
            curve = [np.mean((yv_c - within_part(m_s, Xv, xv, yrv, j)) ** 2) for j in range(len(grid))]
        j = int(np.argmin(curve))
        m = fit_ridge(kind, *rows(train), grid[j:j + 1], b, pooled=pooled)
        Xt, xt, _, yrt = rows(test)
        if log is not None:
            log.append({"model": kind, "input": tag, "pooled": pooled, "test_year": int(yrt[0]),
                        "train_years": f"{years[0]}-{years[-1]}",
                        "alpha": float(grid[j]) if kind != "no text" else np.nan,
                        "line_slope": b, "fitted_slope": float(m["gamma"][0])})
        if pooled:
            return pooled_prediction(m, Xt, xt)
        return np.mean(a + b * xt) + within_part(m, Xt, xt, yrt)
    return predict


# ---------------------------------------------------------------- attention network

def make_attention(df, chunks, seed, label="Item 7", pooling="attention", hp=None, log=None):
    """The attention network of 03 with an unpenalised slope on x, learnt with the text.

    Target: y with each training year's mean removed, scaled. Input: the chunks, and x with each year's mean
    removed, scaled. Prediction: net(chunks) + gamma * x, gamma starting at the no-text within-year slope and
    excluded from weight decay (the joint ridge leaves x unpenalised too). The epoch count is chosen on the last
    training year, then the model is refitted on the whole window from the same seed. Returns a dict of two
    predict functions, "no refit" and "refit", that share one training run per split.
    """
    import torch
    from torch import nn
    import attention

    hp = hp or attention.HP
    dev = attention.DEVICE
    sections = attention.sections_of(label)
    x_all, y_all, yr_all = df["logvol_before"].to_numpy(), df["logvol_after"].to_numpy(), df["year"].to_numpy()
    cache = {}

    class JointNet(nn.Module):
        def __init__(self, gamma0, dim):
            super().__init__()
            self.net = attention.AttentionNet(hp["width"], hp["attention"], hp["head"], hp["dropout"], pooling, dim)
            self.gamma = nn.Parameter(torch.tensor(float(gamma0)))

        def forward(self, x, section, mask, xb):
            out, a = self.net(x, section, mask)
            return out + self.gamma * xb, a

    def run(idx, val_idx, target, xs, stats, gamma0, n_epochs):
        """Train on idx; with val_idx early-stop on it, else run exactly n_epochs. Returns model, history."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = JointNet(gamma0, len(stats[0])).to(dev)
        opt = torch.optim.AdamW([{"params": model.net.parameters(), "weight_decay": hp["weight_decay"]},
                                 {"params": [model.gamma], "weight_decay": 0.0}], lr=hp["lr"])
        loss_fn = nn.MSELoss()
        t_all = torch.from_numpy(target.astype(np.float32)).to(dev)
        x_t = torch.from_numpy(xs.astype(np.float32)).to(dev)
        best, wait, hist = (np.inf, None, -1), 0, []
        for epoch in range(n_epochs):
            model.train()
            t0, total = time.perf_counter(), 0.0
            for bt in attention.batches(idx, hp["batch"], rng):
                opt.zero_grad()
                pred, _ = model(*attention.make_batch(chunks, bt, sections, stats), x_t[bt])
                loss = loss_fn(pred, t_all[bt])
                loss.backward()
                opt.step()
                total += loss.item() * len(bt)
            row = {"epoch": epoch, "train_mse": total / len(idx), "seconds": time.perf_counter() - t0,
                   "gamma": float(model.gamma.detach())}
            if val_idx is not None:
                row["val_mse"] = float(np.mean((predict_z(model, val_idx, xs, stats) - target[val_idx]) ** 2))
                if row["val_mse"] < best[0]:
                    best, wait = (row["val_mse"], copy.deepcopy(model.state_dict()), epoch), 0
                else:
                    wait += 1
            hist.append(row)
            if val_idx is not None and wait >= hp["patience"]:
                break
        if val_idx is not None:
            model.load_state_dict(best[1])
        return model, pd.DataFrame(hist)

    def predict_z(model, idx, xs, stats):
        model.eval()
        x_t = torch.from_numpy(xs.astype(np.float32)).to(dev)
        out = []
        with torch.no_grad():
            for bt in attention.batches(idx, 4 * hp["batch"]):
                out.append(model(*attention.make_batch(chunks, bt, sections, stats), x_t[bt])[0].cpu().numpy())
        return np.concatenate(out)

    def fit_both(train, test):
        key = (tuple(sorted(set(yr_all[train]))), int(yr_all[test][0]))
        if key in cache:
            return cache[key]
        years = key[0]
        fit, val = train & (yr_all != years[-1]), train & (yr_all == years[-1])
        fit_idx, val_idx, train_idx, test_idx = (np.flatnonzero(m) for m in (fit, val, train, test))
        a, b = scoring.fit_line(df, train)
        line_level = np.mean(a + b * x_all[test_idx])
        outs, t0 = {}, time.perf_counter()
        info = {}
        for version, idx, v_idx in [("no refit", fit_idx, val_idx), ("refit", train_idx, None)]:
            # targets and x centred on each year's own mean: training years from training rows only, and the
            # validation and test years on their own (their x and text are known at filing; their y is not used
            # for the test year)
            yc = np.zeros(len(df)); xc = np.zeros(len(df))
            yc[train_idx] = centre(y_all[train_idx], yr_all[train_idx])
            xc[train_idx] = centre(x_all[train_idx], yr_all[train_idx])
            xc[test_idx] = centre(x_all[test_idx], yr_all[test_idx])
            g = (xc[idx] @ yc[idx]) / (xc[idx] @ xc[idx])                # the no-text within-year slope
            scale, sx = (yc[idx] - g * xc[idx]).std(), xc[idx].std()   # scale as in 03: the residual's sd
            target, xs = yc / scale, xc / sx
            gamma0 = g * sx / scale                                       # the same slope in these units
            stats = attention.chunk_stats(chunks, idx, sections)
            n_epochs = hp["max_epochs"] if v_idx is not None else info["best_epoch"] + 1
            model, hist = run(idx, v_idx, target, xs, stats, gamma0, n_epochs)
            if v_idx is not None:
                info["best_epoch"] = int(hist["val_mse"].idxmin())
                info["epochs"] = len(hist)
            z = predict_z(model, test_idx, xs, stats) * scale
            outs[version] = line_level + z - z.mean()                     # centred on the test year
            info[f"gamma_{version}"] = float(model.gamma.detach()) * scale / sx
            info[f"gamma0_{version}"] = gamma0 * scale / sx
        if log is not None:
            log.append({"input": label, "pooling": pooling, "seed": seed, "test_year": key[1],
                        "train_years": f"{years[0]}-{years[-1]}", "line_slope": b, **info,
                        "seconds": time.perf_counter() - t0})
        cache[key] = outs
        return outs

    return {v: (lambda df_, train, test, v=v: fit_both(train, test)[v]) for v in ["no refit", "refit"]}
