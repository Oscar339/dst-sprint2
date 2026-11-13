"""Read every report with FinBERT once and save one 768-number vector per chunk.

Why a script and not a notebook: the full run takes about 11 hours, so it has to
survive the Mac sleeping or a crash. It saves its work in "shards" of 250 reports
and skips any shard that already exists, so re-running it simply carries on.

What it does, per report:
  1. Pick the text: Item 1 and Item 7 from EDGAR-CORPUS; Item 7 from Kogan's
     corpus when EDGAR's is under 100 words (the choice made in
     03-Item7-Source.ipynb, saved in data/processed/item7_source.csv).
  2. Turn the text into FinBERT tokens and cut it into chunks of 510 tokens
     (FinBERT reads at most 512; two slots are the start/end markers).
  3. Keep at most MAX_CHUNKS chunks per section, but record the true number.
  4. Run each chunk through FinBERT and keep the CLS vector (FinBERT's
     summary of the chunk), 768 numbers, saved at half precision.

Usage (from the repo root, with the shared venv):
  python Oscar/extract_finbert.py --smoke          # 50 reports, checks fp16 vs fp32
  python Oscar/extract_finbert.py                  # the full run, all years
  python Oscar/extract_finbert.py --years 2006 --out /some/folder
"""
import argparse
import gzip
import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

MODEL = "ProsusAI/finbert"
CHUNK = 510          # tokens of text per chunk (FinBERT's limit is 512 incl. 2 markers)
MAX_CHUNKS = 64      # keep at most this many chunks per section
BATCH = 32           # chunks through the GPU at once
SHARD = 250          # reports per saved file

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
KOGAN = ROOT / "data" / "raw" / "kogan"
DEFAULT_OUT = Path.home() / "OneDrive - University of Bristol" / "DST" / "finbert"


def load_plan():
    """One row per usable report: key, cik, year, item7_source."""
    src = pd.read_csv(PROCESSED / "item7_source.csv")
    src = src[src["usable"]].copy()
    return src.sort_values(["year", "key"]).reset_index(drop=True)


def load_texts(year, rows):
    """Return {key: (item1_text, item7_text)} for the reports in `rows` (one year)."""
    wanted = {(r.cik, r.year) for r in rows.itertuples()}
    edgar = {}
    with gzip.open(PROCESSED / "edgar_matched" / f"{year}.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            if (rec["cik"], rec["year"]) in wanted:
                edgar[(rec["cik"], rec["year"])] = rec

    need_kogan = set(rows.loc[rows["item7_source"] == "kogan", "key"])
    kogan = {}
    if need_kogan:
        with tarfile.open(KOGAN / f"{year}.tok.tgz") as tar:
            for m in tar:
                if m.isfile():
                    key = Path(m.name).name.split(".")[0]
                    if key in need_kogan:
                        kogan[key] = tar.extractfile(m).read().decode("utf-8", "replace")

    out = {}
    for r in rows.itertuples():
        rec = edgar.get((r.cik, r.year), {})
        item7 = kogan[r.key] if r.item7_source == "kogan" else rec.get("section_7", "")
        out[r.key] = (rec.get("section_1", ""), item7)
    return out


def embed(model, tok, texts, device):
    """Embed each text. Returns (list of (n_kept, 768) float16 arrays, list of true chunk counts)."""
    all_chunks, kept_counts, true_counts = [], [], []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        chunks = [ids[i:i + CHUNK] for i in range(0, len(ids), CHUNK)]
        true_counts.append(len(chunks))
        chunks = chunks[:MAX_CHUNKS]
        kept_counts.append(len(chunks))
        all_chunks.extend(chunks)

    vecs = []
    with torch.inference_mode():
        for i in range(0, len(all_chunks), BATCH):
            batch = tok.pad(
                {"input_ids": [[tok.cls_token_id] + c + [tok.sep_token_id] for c in all_chunks[i:i + BATCH]]},
                return_tensors="pt",
            )
            out = model(**{k: v.to(device) for k, v in batch.items()}).last_hidden_state[:, 0]
            vecs.append(out.float().cpu().numpy().astype(np.float16))
    flat = np.concatenate(vecs) if vecs else np.zeros((0, 768), np.float16)

    per_text, start = [], 0
    for n in kept_counts:
        per_text.append(flat[start:start + n])
        start += n
    return per_text, true_counts


def load_model(device, dtype):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=dtype).to(device).eval()
    return tok, model


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def smoke(n=50):
    """Embed n reports in fp32 and fp16 and compare, so we can trust fp16."""
    device = pick_device()
    plan = load_plan()
    rows = plan[plan["year"] == 2005].head(n)
    texts = load_texts(2005, rows)
    item7 = [texts[k][1] for k in rows["key"]]
    results = {}
    for name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        tok, model = load_model(device, dtype)
        embed(model, tok, item7[:2], device)  # warm-up: first call is slow
        t = time.perf_counter()
        vecs, true = embed(model, tok, item7, device)
        if device == "mps":
            torch.mps.synchronize()
        secs = time.perf_counter() - t
        n_chunks = sum(len(v) for v in vecs)
        print(f"{name}: {n_chunks} chunks in {secs:.1f}s = {n_chunks / secs:.1f} chunks/s on {device}")
        results[name] = np.concatenate(vecs).astype(np.float32)
        del model
    a, b = results["fp32"], results["fp16"]
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    print(f"fp16 vs fp32 cosine similarity: mean {cos.mean():.5f}, worst {cos.min():.5f}")
    print(f"largest absolute difference in any number: {np.abs(a - b).max():.4f} (typical size of a number: {np.abs(a).mean():.3f})")
    print(f"truncated at {MAX_CHUNKS} chunks: {np.mean(np.array(true) > MAX_CHUNKS):.1%} of these reports")


def run(years, out_dir, dtype_name):
    device = pick_device()
    dtype = torch.float16 if dtype_name == "fp16" else torch.float32
    out_dir.mkdir(parents=True, exist_ok=True)
    revision = Path(snapshot_download(MODEL, local_files_only=True)).name
    manifest = {
        "model": MODEL, "model_revision": revision, "dtype": dtype_name, "chunk_tokens": CHUNK,
        "max_chunks": MAX_CHUNKS, "vector": "CLS (last_hidden_state[:, 0])", "batch": BATCH,
        "shard_size": SHARD, "torch": torch.__version__, "device": device,
        "item7_source": "data/processed/item7_source.csv (EDGAR, Kogan if EDGAR < 100 words)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    tok, model = load_model(device, dtype)
    plan = load_plan()
    t_start, done_chunks = time.perf_counter(), 0
    for year in years:
        rows_y = plan[plan["year"] == year]
        n_shards = -(-len(rows_y) // SHARD)
        texts = None  # loaded lazily, only if a shard of this year is missing
        for s in range(n_shards):
            path = out_dir / f"{year}_{s:03d}.npz"
            if path.exists():
                continue
            if texts is None:
                texts = load_texts(year, rows_y)
            rows = rows_y.iloc[s * SHARD:(s + 1) * SHARD]
            keys = list(rows["key"])
            result = {"keys": np.array(keys), "item7_source": rows["item7_source"].to_numpy().astype(str)}
            for name, idx in [("item1", 0), ("item7", 1)]:
                vecs, true = embed(model, tok, [texts[k][idx] for k in keys], device)
                result[f"{name}_n"] = np.array([len(v) for v in vecs])
                result[f"{name}_true"] = np.array(true)
                result[f"{name}_emb"] = np.concatenate(vecs)
                done_chunks += len(result[f"{name}_emb"])
            tmp = out_dir / f"{year}_{s:03d}.tmp.npz"
            np.savez(tmp, **result)
            os.replace(tmp, path)  # only a finished shard ever has the real name
            rate = done_chunks / (time.perf_counter() - t_start)
            print(f"{year} shard {s + 1}/{n_shards} saved  ({rate:.1f} chunks/s overall)", flush=True)
    print("all requested shards present")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="small check of fp16 vs fp32, then stop")
    p.add_argument("--years", default="2006,2005,2004,2003,2002,2001,2000,1999,1998,1997,1996",
                   help="comma-separated, processed in this order")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    a = p.parse_args()
    if a.smoke:
        smoke()
    else:
        run([int(y) for y in a.years.split(",")], a.out, a.dtype)
