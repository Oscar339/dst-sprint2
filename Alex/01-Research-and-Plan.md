# Models: research and plan

In this section, we look at what the brief asks of our models, how to score them fairly, which linear model to
start with and how a report goes into it. It ends with what we decided to build, which is carried out in
`02-Models.ipynb`, and an updated plan from what the first models showed.

Oscar has the data ready (`report/02-Data.ipynb`): 27,142 reports, each section turned into FinBERT chunk vectors,
with the shared training and test years. Louis is starting on an MLP. My part is how every model is scored, the
simplest text model (a linear one), and then the main model, the attention network.

## 1. What the brief asks

The brief marks four things: (a) the implementation, with credit for going beyond off-the-shelf, exploring
multiple implementations and examining the maths of our choices; (b) the application; (c) the correctness of the
methods; and (d) the robustness of the results. It also says we need to *"perform robustly in all areas and
additionally demonstrate insight somewhere"*.

For the models this means:

* Every model is scored the same way, on several test years, with a measure of uncertainty. One test year with no
  interval is weak on (c) and (d), however good the model.
* A ladder of models, each adding one thing, so any gain can be traced to its cause. If a neural network beats the
  baseline, we need to know whether the network helped or just the text.
* The maths of each choice is written down.

## 2. What we predict and how it is scored

We follow Kogan et al. (2009), who built the dataset:

* **Target** - `logvol_after`, the log of the standard deviation of daily returns over the 12 months after filing.
* **Score** - mean squared error (MSE) on a held-out year, for test years 2001–2006, each trained on the five
  years before.
* **Their baseline** - simply copying last year's log volatility. Their only fitted model on last year's
  volatility alone (support vector regression) did worse than copying (Table 2: an MSE of 5.68 in 2006 with an
  intercept), so the paper has no working fitted history-only baseline.
* **Their result** (Table 2) - text plus last year's volatility beat copying in four of the six test years
  (2003–06), for 2006 an MSE of 0.1402 against 0.1463, but not in 2001 or 2002.

We score on log volatility rather than volatility itself. Volatility is strongly right-skewed, so squared errors
on it would be dominated by the few most volatile companies. On the log scale the distribution is close to
symmetric, "twice as volatile" counts the same as "half as volatile", and we can compare with the paper directly.
A classification version ("will this company be in next year's riskiest 10%?") is a possible extra later, but it
throws away information and loses the comparison with the paper.

## 3. Which years to train on

A report's label covers the 12 months after filing, so a report filed in December 2005 has a label running into
2006. In real use that label is not known yet when a 2006 report arrives, so training on it uses information from
the future. Oscar therefore fixed two windows in `report/02` (saved in `data/output/splits.csv`), and every model
in the project uses them:

| Window | Years for test year $T$ | Pros | Cons |
|---|---|---|---|
| paper | $T-5 \dots T-1$ | Same as Kogan et al., so directly comparable | Last year's labels overlap the test year |
| strict | $T-5 \dots T-2$ | No overlap with the test year | One year less data |

`report/02` also sets the rule for hyperparameters: they are tuned on the last training year, never on the test
year. For the neural networks, that year is also the one used for early stopping.

## 4. Telling whether a difference is real

Two models' MSEs on the same test year always differ a little by chance, so we need an interval on the
difference. For test report $i$, let $d_i = (y_i - \hat y_i^{\text{model}})^2 - (y_i - \hat y_i^{\text{base}})^2$,
so the difference in MSE is $\bar d$. A paired bootstrap resamples the test reports many times and recomputes
$\bar d$ (Efron & Tibshirani, 1993). Both models are scored on the same resampled reports, so the noise from
which reports happen to be in the test year cancels. This is the same idea as the Diebold–Mariano test for
comparing forecasts (Diebold & Mariano, 1995).

The bootstrap treats reports as independent, but reports in the same year share the market's shocks - in a crash
year everyone's volatility rises together. So each interval is somewhat too narrow, and the stronger evidence is
consistency across the six test years.

## 5. A linear baseline for the neural networks

The project is about neural networks, but they need a bar to beat that uses the same inputs. If a linear model on
the averaged FinBERT vectors does as well as an MLP, the hidden layers added nothing. A linear model is also a
neural network with no hidden layer, so it is the bottom rung of our ladder: ridge, then Louis's MLP, then the
attention network.

We use ridge regression. The inputs are 768 numbers per section (1,536 for both), heavily correlated with each
other, so ordinary least squares would mostly fit noise. Lasso assumes only a few inputs matter, which does not
suit FinBERT's dimensions, as none of them means anything on its own. Ridge keeps every input but shrinks them:
$$\hat\beta_\lambda = (X^\top X + \lambda I)^{-1} X^\top y, \qquad X\hat\beta_\lambda = \sum_j u_j \, \frac{d_j^2}{d_j^2 + \lambda} \, u_j^\top y,$$
where $X = U D V^\top$. Directions with a lot of spread across reports are kept almost whole, and those with
little spread, where the fit would chase noise, are shrunk towards zero (Hastie, Tibshirani & Friedman, 2009,
§3.4.1). The inputs are standardised on the training rows, and $\lambda$ is chosen on the last training year
(section 3): fit on the other years, score the last one, keep the best $\lambda$ and refit on the whole window.

## 6. How a report goes into the model

A report has a variable number of chunk vectors (median 32), and a linear model needs a fixed length:

| Option | Pros | Cons |
|---|---|---|
| **Average the chunks** (the `X` table in `data.pkl`) | Simple, uses every chunk | A few important chunks get drowned out |
| Maximum over chunks | Keeps strong signals | Sensitive to one odd chunk |
| First chunk only | Cheapest | Ignores most of the report |
| Learned weights (attention) | Can focus on the chunks about risk | Needs a neural network - our main model |

Averaging is exactly the attention network with all weights equal, so comparing the two measures what learning
the weights buys.

Last year's volatility goes in through two stages rather than as one more input to the ridge:

1. fit the straight line $\hat y^{\text{line}} = a + b \cdot \text{logvol\_before}$ (the baseline);
2. fit ridge on the text to the residual $y - \hat y^{\text{line}}$, the part of next year's volatility that last
   year's does not explain.

This way the baseline coefficient is not penalised, and "ridge against the line" measures what the text adds. We
fit Item 1 only, Item 7 only and both.

One weakness to watch is the encoder. The vectors are the CLS vector of `ProsusAI/finbert`, which was further
trained to label financial sentences as positive, negative or neutral (Araci, 2019), so it may mostly encode tone.
If ridge finds little signal, this is the first suspect, and the fix is the FinBERT of Yang et al. (2020), trained
on 10-Ks, earnings calls and analyst reports. FETILDA (Xia et al., 2022) compared the two, but on banks' return on
assets rather than volatility: Yang's version won by only about 4% (their Table 6), so this is weak evidence. They
also found the CLS vector beat averaging the token vectors (Table 7), which supports Oscar's choice, and that
fine-tuning FinBERT beat keeping it frozen by about 5% on volatility.

## 7. What we decided

| Decision | Choice | Why |
|---|---|---|
| Target and score | MSE on log volatility, test years 2001–2006 | Matches Kogan et al., and logs stop the most volatile companies dominating (section 2) |
| Baselines | Copy last year, and a straight line fitted on it | Copying is the paper's baseline; the line uses last year's volatility as well as a one-number rule can, so it is the bar the text has to clear |
| Training windows | paper and strict, from `data/output/splits.csv` | Oscar's windows: comparable with the paper, and one with no look-ahead (section 3) |
| Uncertainty | Paired bootstrap per test year, and consistency across years | Section 4 |
| Linear baseline for the networks | Ridge on the averaged chunk vectors, fitted to the line's residual | Sections 5 and 6 |
| Inputs | Item 1, Item 7 and both | Which section carries the risk information |
| Choosing $\lambda$ | On the last training year | `report/02`'s rule, the same one the networks use for early stopping |
| Shared code | `scoring.py`: every model is a function `predict(df, train, test)` | So Louis's MLP and the attention network are scored exactly the same way |
| Then | The attention network, or a second encoder if ridge finds nothing | Ridge clearly beating the line means the vectors are useful; if not, the vectors are the bottleneck |

To score a model with the shared code:

```python
import pickle, sys; sys.path.insert(0, "../Alex")
import scoring
with open("../data/processed/data.pkl", "rb") as f:
    d = pickle.load(f)
df = scoring.frame(d)
results = scoring.evaluate(df, predict, name="MLP")
```

## 8. Updated plan (14 November), after the linear models

What the first models showed (`02-Models.ipynb`, section 5):

* Most of the error is the year's overall volatility level, which no single report can know. A model fitted on
  several years together can use the text to guess the era, which looks like big gains and losses that are really
  about the year. Every model now reports its error split into the level and the spread (how well it ranks
  companies within a year).
* Within a year, the averaged FinBERT vectors add little: −6% to +7% by year, −1.0% to +0.3% on average, and in
  11 of 72 fits the last training year chose to switch the text off. By the rule above, the averaged vectors look
  like the weak point.
* On Kogan et al.'s own full data and windows, a straight line on last year's volatility, with no text, scores
  0.1403 for 2006 against their text model's 0.1402. Much of the paper's gain comes from fitting history, not from
  the text.
* The 11% of reports not matched to EDGAR-CORPUS are the hard cases: copying last year gives 3–6 times the error
  there. So our MSEs are lower than the paper's for reasons that have nothing to do with any model.

Next steps (content freeze Monday 30 November, 12:00):

| When | What | Who |
|---|---|---|
| Group | Agree: report against both copying and the line, with the line as the bar; whether the second FinBERT goes back in | all |
| 16–18 Nov | Attention network in `02-Models.ipynb`: the chunk vectors from `data.pkl`, attention pooling with the padding masked (Ilse et al., 2018), predicting the within-year residual over the line; dropout, early stopping on the last training year and 5 seeds, on a Colab GPU; paper and strict windows; plots of the attention weights | Alex |
| 16–18 Nov | MLP, scored with `scoring.evaluate` against the within-year ridge | Louis |
| 19–20 Nov | If attention is no better than the within-year ridge: embed Item 7 with Yang et al.'s FinBERT on Colab, timed on CPU and GPU across batch sizes (this doubles as the scaling experiment) | Alex, Oscar |
| Optional | Embed Kogan's own Item 7 for the ~2,000 dropped reports, so Item 7 models can be scored on the paper's full test sets | Oscar |
| 23–27 Nov | Scaling write-up: map/reduce for FinBERT and for the per-year ridge sums, with the timings | Alex, Oscar |
| By 30 Nov | Models notebook into `report/03-Models.ipynb`; every notebook rerun top to bottom with outputs | all |

## References

- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv:1908.10063*.
- Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., §3.4 (shrinkage methods). Springer.
- Ilse, M., Tomczak, J. M. & Welling, M. (2018). Attention-based deep multiple instance learning. *ICML 2018*.
- Kogan, S., Levin, D., Routledge, B. R., Sagi, J. S. & Smith, N. A. (2009). Predicting risk from financial reports with regression. *NAACL-HLT 2009*, 272–280. https://aclanthology.org/N09-1031/
- Xia, B., Rawte, V. D., Zaki, M. J. & Gupta, A. (2022). FETILDA: An effective framework for fin-tuned embeddings for long financial text documents. *arXiv:2206.06952*.
- Yang, Y., Uy, M. C. S. & Huang, A. (2020). FinBERT: A pretrained language model for financial communications. *arXiv:2006.08097*.
