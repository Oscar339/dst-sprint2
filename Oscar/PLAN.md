# Sprint 2 plan: predicting company risk from annual reports

**Our agreed project plan (Oscar, Alex, Louis).** Owners and equity still to be confirmed.
Deadline: Wednesday 2 December, 12:00. Group content final by **Monday 30 November, 12:00**.

## 1. Goal

**Question:** does the text of a company's annual report predict how volatile
its share price will be over the next year, beyond what last year's volatility
already tells us?

- **Volatility** measures how much a share price jumps around. It is the
  standard measure of investment risk.
- Last year's volatility is already a strong predictor, so it is the
  **baseline** every model has to beat.
- The brief asks us to (a) choose a question, (b) measure how the model
  performs computationally, and (c) apply it. It must run on a laptop or
  Colab, and we must argue how it would cope with **1000× more data**.

## 2. The data

**The 10-K Corpus** (Kogan et al., 2009), free from
[cs.cmu.edu/~ark/10K](https://www.cs.cmu.edu/~ark/10K). No login, about 350 MB.

- About **29,000 annual reports** ("10-Ks") from US public companies, 1996–2006.
- For each report we use two sections:
  - **Item 7, management's discussion of the year** (MD&A), typically about
    10,000 words, included in the corpus already split into words
  - **Item 1, the business description** (what the company does, its
    markets, competitors and customers), typically about 4,900 words. This
    comes from **EDGAR-CORPUS** (Loukas et al., 2021), which has every 10-K
    already split into sections. We match it to our reports by SEC company
    number and year. In a 2006 test about 97% matched; other years still need
    checking.
- Already attached to each report:
  - **volatility in the 12 months before** the filing (the baseline)
  - **volatility in the 12 months after** the filing (what we predict)
  - filing date, company name and SEC company number
- The volatilities were computed from **CRSP**, the standard academic price
  database. We don't download any prices ourselves.
- The original paper reports scores on this exact data, so we can compare our
  results with theirs directly.

**Known limitations** (we discuss these in the report, not hide them):

- The data is from 1996–2006, not today.
- Companies that were delisted in the year after filing have no label (under 4%).
- No "Risk Factors" section (Item 1A). Companies only had to include it from
  late 2005, so it is missing for almost all our labelled years (1% of 2004
  filings, 53% of 2005, 96% of 2006).
- Some reports won't match in EDGAR-CORPUS for Item 1; we report how many.

**Other sources we considered** (to go in the data-search log, which the brief rewards):

| Source | Why not chosen |
|---|---|
| EDGAR-CORPUS (Hugging Face) | **Used for Item 1 only.** Every 10-K 1993–2020 (40 GB) but no prices, so not usable as the main dataset |
| Stocks-Yearly-Sections10K (Hugging Face) | Every 10-K section since 2001 with tickers, but downloads need a paid subscription, and no prices |
| Yahoo Finance prices | Mostly missing companies that no longer exist (survivorship bias) |
| Bloomberg Terminal | Only Oscar has access, in person; the data can't go in a public repo |
| NCCU extended 10-K corpus (1996–2013) | Website security certificate has expired; worth retrying |

## 3. How the model works

We **use** someone else's model to read the reports, and we **build and
train our own** neural network to predict volatility from what it read.

```
Annual report: Item 1 + Item 7 (~15,000 words)
        │
        ▼
 Cut into ~35 chunks
        │
        ▼
┌─────────────────────┐
│ FinBERT             │  ← SOMEONE ELSE'S. Downloaded, not changed.
│ (reads each chunk)  │     Turns each chunk into 768 numbers.
└─────────────────────┘
        │
        ▼
 ~35 lists of 768 numbers + last year's volatility
        │
        ▼
┌─────────────────────┐
│ Our neural network  │  ← OURS. We write it in PyTorch and train it,
│ (decides which      │     like the week 7 lab.
│  chunks matter,     │
│  predicts risk)     │
└─────────────────────┘
        │
        ▼
 Predicted volatility for next year
```

- **FinBERT** is BERT (Google's language model) further trained on financial
  text. It can only read about 400 words at a time, which is why we cut each
  report into chunks. Reusing a trained model like this is **transfer
  learning**, which the brief lists as an approved approach.
- **Our network** combines the ~35 chunk summaries into one per report, using
  an **attention** layer that learns which chunks matter most, then predicts
  volatility. Most of the implementation marks come from this part.

### Models we compare

| Model | Neural? | Built by us? | Purpose |
|---|---|---|---|
| Copy last year's volatility | no | – | Simplest baseline |
| Straight line from last year's volatility | no | – | Fitted baseline |
| Word counts (TF-IDF) + support vector regression | no | uses scikit-learn | Reproduces the 2009 paper |
| Finance word lists (Loughran & McDonald, 2011) | no | uses scikit-learn | Simple, explainable text baseline |
| Small MLP on word counts | **yes** | **yes**, from scratch | Neural network without FinBERT |
| FinBERT + averaging + regression | partly | partly | Is FinBERT's reading useful at all? |
| FinBERT + **our attention network** | **yes** | **yes** | **Main model** |
| A second FinBERT (Yang et al., 2020, trained on company reports) | yes | partly | Which pre-trained model reads better? |
| Fine-tuned FinBERT (stretch goal) | yes | partly | Adjust FinBERT itself; needs a Colab GPU |

### Which part of the report carries the risk information?

We run the main models three ways: **Item 7 only**, **Item 1 only**, and
**both**. That answers a simple, clear question for the report: does the
business description or management's discussion say more about future risk?
It also shows whether the attention network learns to focus on one section.

### Stretch goal: a modern check with Risk Factors

The main study stays on 1996–2006. If time allows, we also test the best
models on **recent filings (e.g. 2016–2019) that include Risk Factors**:

- Text (Items 1, 1A and 7) from EDGAR-CORPUS.
- Volatility **calculated by us** from daily Yahoo Finance prices (`yfinance`),
  as the standard deviation of daily returns over the 12 months after filing,
  the same definition as the 2009 paper.
- Companies are matched to tickers with the SEC's company list.
- **Known weakness:** Yahoo and the SEC list mostly cover companies still listed
  today, so companies that failed or were bought drop out (survivorship bias).
  We measure how many drop out and say so. A test on 517 filings from 2018
  (10 Nov) found only **44% usable** with Yahoo prices: 81% stated their
  ticker in the report, and only 55% of those tickers still had Yahoo prices
  (missing ones include bankrupt Bed Bath & Beyond and acquired Anadarko).
  So with Yahoo this check is heavily biased. If the university has **WRDS**
  (which provides CRSP, the same price source as the main labels), we use that
  instead and the problem goes away. Ask the library.
- This answers two of the brief's points: does the model still work on recent
  data, and how does our dataset compare with a real-world setting?

### How we score models

- **Train on earlier years, test on a later year**, never mixing them, so a
  model never sees the future.
- Main score: **mean squared error (MSE) on log volatility**, the paper's measure.
- Check several test years (2003–2006), with confidence intervals.
- Two training windows: the paper's, and a stricter one that stops a year
  earlier so no training label overlaps the test year.

### The "scale" part

- Time FinBERT on CPU vs Mac GPU vs Colab GPU, with different batch sizes and
  data sizes.
- Argument for 1000× more data: each chunk is processed **independently**, so
  the work splits across many GPUs (map), then the results are combined per
  report (reduce).
- 1000× this corpus (about 30 million reports) is more 10-Ks than exist, so the
  argument must cover a wider stream: all SEC filing types, earnings-call
  transcripts and news.

## 4. What we expect

From quick test runs on Oscar's Mac (7–10 Nov). The project re-measures all of these in its own notebooks:

| Result | Expectation | How sure |
|---|---|---|
| Our setup reproduces the paper | Baseline 0.1464 vs paper 0.1463 for 2006 | **Measured** |
| Text alone vs last year's volatility | Text alone is worse | High: seen here and in the paper |
| Text + last year's volatility (word counts) | 2–25% better depending on year; 2005–06 gains are small and, with the stricter window, could be chance | **Measured** |
| FinBERT vs word counts | Roughly level, could go either way | **Guess, untested** |
| Our attention network | Needs tuning; overfit badly in a small first test | Measured on a small sample only |
| Speed | FinBERT about 27 chunks/second on the Mac GPU, about 3× the CPU; time grows in proportion to data | **Measured** |
| Time to read ~7,000 reports with FinBERT | About 3 hours on the Mac for Item 7, about 4.5 hours with Item 1 added; done once, then everyone reuses the saved output | Estimated from the measured speed |
| Item 1 vs Item 7 | Unknown; MD&A is the section the paper used, so it may carry more | **Untested** |

**We should agree now** that "the neural network didn't beat the simple model"
is an acceptable result. The brief rewards correct methods and honest
conclusions, not a big improvement.

## 5. Task list

Owners are suggestions; to be agreed with equity.

### Week of 9 Nov: agree and set up
- [ ] Agree the topic, the split of work and equity (all)
- [ ] Add everyone's names and equity to the README (all)
- [ ] `report/01-Data.ipynb`: download the data, remove the duplicate reports, explore it, make the plots (volatility before vs after, report length over time) (**Data**)
- [ ] Download Item 1 from EDGAR-CORPUS for the years we use, match it to our reports, and record how many match each year (**Data**)
- [ ] Write the data-search log in `data/README.md` (**Data**)
- [ ] Read FinBERT over ~7,000 reports (2004–06), Item 1 and Item 7 kept separate, once; save the output to OneDrive, not GitHub (**Neural**)
- [ ] Start the reading list: Kogan 2009, Loughran & McDonald 2011, Araci 2019, Yang 2020, EDGAR-CORPUS 2021, FETILDA 2022, Engle 1982 (all)

### Week of 16 Nov: models
- [ ] Baselines: copy, straight line, word counts + support vector regression, finance word lists (**Data**)
- [ ] Small MLP on word counts, from scratch (**Neural**)
- [ ] FinBERT + averaging (**Neural**)
- [ ] Our attention network, with dropout, early stopping and several random seeds (**Neural**)
- [ ] Try the second FinBERT (**Neural**)
- [ ] Timing experiments: CPU vs GPU, batch size, data size (**Scale**)

### Week of 23 Nov: robustness and scale
- [ ] Score every model on test years 2003–06, both training windows, with confidence intervals (**Data**)
- [ ] Compare Item 1 only vs Item 7 only vs both (**Data** + **Neural**)
- [ ] Plot which chunks the attention network focuses on (**Neural**)
- [ ] Write the 1000× argument, with the timing plots (**Scale**)
- [ ] Stretch: fine-tune FinBERT on Colab (**Neural** + **Scale**)
- [ ] Stretch: modern check on 2016–19 filings with Risk Factors; Yahoo prices, or CRSP via WRDS if available (**Data** + **Neural**)
- [ ] Ask the library whether Bristol has WRDS / CRSP (anyone, this week)
- [ ] Draft `report/` notebooks: introduction, data, models, results, scale, conclusion (all)

### By Monday 30 Nov, 12:00: finish
- [ ] Run every notebook top to bottom, then commit with outputs showing (all)
- [ ] README: reading order, install steps, equity, any manual steps (such as downloading the saved FinBERT output) (all)
- [ ] Check every source is cited (all)
- [ ] Each pick a **different** maths topic for our reflections, e.g. attention/pooling; support vector regression and regularisation; volatility and ARCH effects (all)

### By Wednesday 2 Dec, 12:00
- [ ] Each submit our individual reflection (500–800 words, written alone) + the repo link on Blackboard

## 6. Risks

| Risk | What we do about it |
|---|---|
| FinBERT adds nothing over word counts | Report it honestly; compare it with the second FinBERT |
| Our attention network overfits | More training data, dropout, early stopping, averaging over seeds |
| FinBERT reading takes too long | Do it once, early, and share the output; use Colab if needed |
| Old data (1996–2006) | Discuss how it compares with today's filings in the report |
| Three weeks is tight | Fine-tuning and the modern check are stretch goals only; the core plan works without them |
| Modern check is biased towards survivors | Measure and report the drop-out; use CRSP via WRDS if available |

## References (start of list)

- Kogan, Levin, Routledge, Sagi & Smith (2009). *Predicting Risk from Financial Reports with Regression*. NAACL. https://aclanthology.org/N09-1031/
- Loughran & McDonald (2011). *When Is a Liability Not a Liability?* Journal of Finance 66(1). https://papers.ssrn.com/abstract=1331573
- Araci (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. https://arxiv.org/abs/1908.10063
- Yang, Uy & Huang (2020). *FinBERT: A Pretrained Language Model for Financial Communications*. https://arxiv.org/abs/2006.08097
- Loukas et al. (2021). *EDGAR-CORPUS*. https://aclanthology.org/2021.econlp-1.2
- FETILDA (2022). Chunk-based embeddings for long financial documents. https://arxiv.org/abs/2206.06952
