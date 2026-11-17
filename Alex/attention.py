"""The attention network of 03-Attention-network.ipynb, as a module, so other notebooks can reuse it unchanged.

The only difference from notebook 03 is that the chunk vectors are passed in (`chunks`: a dict from section name to
a list with one (n_chunks x 768) array per report, in the same order as the rows of `df`), instead of being read
from data.pkl, so the same network can be trained on vectors from a different encoder.

    import attention
    predict = attention.make_attention(df, chunks, "Item 7", seed=0)
    results = scoring.evaluate(df, predict, name="attention pooling, Item 7, seed 0")
"""
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import scoring

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SECTION_ID = {"Item 1": 0, "Item 7": 1}
HP = dict(width=128, attention=64, head=64, dropout=0.2, lr=1e-3, weight_decay=1e-2,
          batch=64, max_epochs=30, patience=5, seeds=[0, 1, 2, 3, 4])


class AttentionNet(nn.Module):
    def __init__(self, width=128, attention=64, head=64, dropout=0.2, pooling="attention", dim=768):
        super().__init__()
        self.pooling = pooling
        self.project = nn.Sequential(nn.Linear(dim, width), nn.ReLU(), nn.Dropout(dropout))
        self.section = nn.Embedding(2, width)
        self.attn_V = nn.Linear(width, attention)
        self.attn_w = nn.Linear(attention, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(width, head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head, 1))

    def forward(self, x, section, mask):
        # x (B, L, dim) standardised chunk vectors; section (B, L) 0/1; mask (B, L) True where there is a chunk
        h = self.project(x) + self.section(section)
        if self.pooling == "attention":
            score = self.attn_w(torch.tanh(self.attn_V(h))).squeeze(-1)
        else:                                                       # mean pooling: every chunk scores the same
            score = torch.zeros(h.shape[:2], device=h.device)
        score = score.masked_fill(~mask, float("-inf"))
        a = torch.softmax(score, dim=1)                             # (B, L), sums to 1 over the real chunks
        z = (a.unsqueeze(-1) * h).sum(1)
        return self.head(z).squeeze(-1), a


def sections_of(label):
    return ["Item 1", "Item 7"] if label == "both" else [label]


def chunk_stats(chunks, idx, sections):
    """Mean and sd per dimension over every chunk of the reports in idx."""
    dim = next(len(c[0]) for c in chunks[sections[0]] if len(c))
    s, ss, n = np.zeros(dim), np.zeros(dim), 0
    for sec in sections:
        for i in idx:
            c = chunks[sec][i].astype(np.float32)
            s += c.sum(0); ss += (c * c).sum(0); n += len(c)
    mu = s / n
    return mu.astype(np.float32), np.sqrt(np.maximum(ss / n - mu * mu, 1e-8)).astype(np.float32)


def make_batch(chunks, idx, sections, stats):
    mu, sd = stats
    seqs = [np.concatenate([chunks[sec][i] for sec in sections]).astype(np.float32) for i in idx]
    secs = [np.concatenate([np.full(len(chunks[sec][i]), SECTION_ID[sec]) for sec in sections]) for i in idx]
    L, dim = max(len(q) for q in seqs), len(mu)
    x = np.zeros((len(idx), L, dim), np.float32)
    section = np.zeros((len(idx), L), np.int64)
    mask = np.zeros((len(idx), L), bool)
    for k, (q, sc) in enumerate(zip(seqs, secs)):
        x[k, :len(q)] = (q - mu) / sd
        section[k, :len(q)] = sc
        mask[k, :len(q)] = True
    return (torch.from_numpy(x).to(DEVICE), torch.from_numpy(section).to(DEVICE), torch.from_numpy(mask).to(DEVICE))


def batches(idx, size, rng=None):
    order = rng.permutation(idx) if rng is not None else np.asarray(idx)
    for i in range(0, len(order), size):
        yield order[i:i + size]


def predict_text(model, chunks, idx, sections, stats, hp=HP):
    """The network's part of the prediction, in standardised residual units."""
    model.eval()
    out = []
    with torch.no_grad():
        for b in batches(idx, 4 * hp["batch"]):
            out.append(model(*make_batch(chunks, b, sections, stats))[0].cpu().numpy())
    return np.concatenate(out)


def train_one(chunks, fit_idx, val_idx, sections, target, stats, seed, hp=HP, pooling="attention"):
    """Fit on fit_idx, keep the epoch with the best MSE on val_idx. Returns the model and its history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AttentionNet(hp["width"], hp["attention"], hp["head"], hp["dropout"], pooling, dim=len(stats[0])).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.MSELoss()
    t_all = torch.from_numpy(target.astype(np.float32)).to(DEVICE)
    best, wait, history = (np.inf, None, -1), 0, []
    for epoch in range(hp["max_epochs"]):
        model.train()
        t0, total = time.perf_counter(), 0.0
        for b in batches(fit_idx, hp["batch"], rng):
            opt.zero_grad()
            pred, _ = model(*make_batch(chunks, b, sections, stats))
            loss = loss_fn(pred, t_all[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        seconds = time.perf_counter() - t0
        val_mse = float(np.mean((predict_text(model, chunks, val_idx, sections, stats, hp) - target[val_idx]) ** 2))
        history.append({"epoch": epoch, "train_mse": total / len(fit_idx), "val_mse": val_mse, "seconds": seconds})
        if val_mse < best[0]:
            best, wait = (val_mse, copy.deepcopy(model.state_dict()), epoch), 0
        else:
            wait += 1
            if wait >= hp["patience"]:
                break
    model.load_state_dict(best[1])
    return model, pd.DataFrame(history)


def make_attention(df, chunks, label, seed, pooling="attention", hp=HP, run_log=None, checkpoint_dir=None, tag=""):
    """A predict(df, train, test) for scoring.evaluate: line + within-year attention network, as in notebook 03."""
    sections = sections_of(label)
    x_all, y_all, yr_all = df["logvol_before"].to_numpy(), df["logvol_after"].to_numpy(), df["year"].to_numpy()

    def predict(df_, train, test):
        years = sorted(set(yr_all[train]))
        fit, val = train & (yr_all != years[-1]), train & (yr_all == years[-1])
        fit_idx, val_idx, test_idx = np.flatnonzero(fit), np.flatnonzero(val), np.flatnonzero(test)
        a, b = scoring.fit_line(df_, train)
        resid = y_all - (a + b * x_all)
        resid = resid - pd.Series(resid).groupby(yr_all).transform("mean").where(train, 0).to_numpy()
        scale = resid[fit_idx].std()
        stats = chunk_stats(chunks, fit_idx, sections)
        t0 = time.perf_counter()
        model, history = train_one(chunks, fit_idx, val_idx, sections, resid / scale, stats, seed, hp, pooling)
        text = predict_text(model, chunks, test_idx, sections, stats, hp) * scale
        text = text - text.mean()                                   # centre on the test year
        if run_log is not None:
            run_log.append({"vectors": tag, "input": label, "pooling": pooling, "seed": seed,
                            "test_year": int(yr_all[test_idx[0]]), "train_years": f"{years[0]}-{years[-1]}",
                            "fit_reports": len(fit_idx), "epochs": len(history),
                            "best_epoch": int(history["val_mse"].idxmin()),
                            "best_val_mse": float(history["val_mse"].min() * scale ** 2),
                            "seconds": time.perf_counter() - t0,
                            "seconds_per_epoch": float(history["seconds"].mean())})
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(checkpoint_dir) /
                       f"{tag}_{label.replace(' ', '')}_{pooling}_{years[0]}-{years[-1]}_seed{seed}.pt")
        return a + b * x_all[test_idx] + text
    return predict
