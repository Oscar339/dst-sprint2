Intro

In this report we investigate whether the text extracted from a company's 10-K filing can be used to effectively predict the volatility of that company's share price, beyond what we could predict from last year's volatility alone.

Every company listed in the US must file a 10-K with the Securities and Exchange Commission (SEC) each year. It is long (the median report in our data runs to about 12,000 words across the two sections we use) and it is written partly to warn investors about what could go wrong. If that warning carries real information, a model that reads the report should predict risk better than one that doesn't.

In this project we will measure risk by volatility, the industry standard for how 'risky' a stock is. Being able to accurately predict volatility has critical implications for option pricing (via Black-Scholes; \citealp{black1973}), risk management, regulation and broader portfolio construction. Given the sheer scale of capital that volatility forecasts are used to guide, the cost of running a model like ours is greatly outweighed by the value of even a small improvement in forecast accuracy. 

The volatility of a stock is nothing more than the standard deviation of its returns. We formalise this with the notation below:

1. $p_t$ denotes the closing price on trading day $t$.
2. $r_t$ is the log return given by

$$
r_t = \log\frac{p_t}{p_{t-1}}.
$$

We then define the volatility $\sigma$ by:

$$
\sigma = \sqrt{\frac{1}{T}\sum_{t=1}^{T}\left(r_t - \bar{r}\right)^2},
\qquad \bar{r} = \frac{1}{T}\sum_{t=1}^{T} r_t
$$
where T is the total number of trading days in a year, typically 252. We then write
$$
y = \log \sigma\text{.}
$$
Here $y$ is called the log volatility. We use both the log volatility and the log returns as these are often better behaved than their standard equivalents. For example, the volatilities of companies are right-skewed, and taking the log turns this into a distribution much closer to normal \citep{andersen2001}. They also carry the benefit of allowing us to compare our results directly to the paper this report is inspired by \citep{kogan2009}.

As a rule of thumb, the volatility of a company's returns are relatively consistent each year \citep{engle1982}. This is because 'risky' stocks tend to have idiosyncratic properties that persist over time that make them more prone to risk than other stocks. For example, the volatility of NVIDIA will almost always be higher than that of PepsiCo. This is because people will always drink Pepsi but AI infrastructure demand remains a hot topic of debate.

With this in mind, predicting the next year's volatility for a company based on just the volatility of previous years is a simple and effective method. We will use this as the baseline and investigate how adding information about a company's SEC filings changes prediction accuracy. Following Kogan et al. (2009), we predict the log of volatility and score models by their mean squared error (MSE).

We use about 27,000 10-K reports from 1996 to 2006. We start by building a ridge regression, the simplist model we can build that looks at the text. Then we build an attention network and a multilayer perceptron (MLP) in an attempt to beat this baseline. We also experiment with different encoders to determine the effect of the encoder's pre-training on text understanding.

Its worth noting hre that even a negative result is informative. If we find that including the text of the 10-K filings does not improve prediction accuracy then this implies something potentially more interesting. It would suggest that the information in the report is already reflected in how the share price has moved, and so is already contained in last year's volatility. This is what one would expect in an efficient market, where the price of a stock should reflect all publicly available information \citep{fama1970}. This is a theme that we will return to throughout the report as the text improved on the baseline only slightly, even in our best models. **[TODO: add numbers once models are trained]**

data:

When sourcing data for the project, we need reliable sources for both the 10-K reports and the share price volatility in the year after they were filed. The 10-K reports are easy enough to source; every report is public on EDGAR, the SEC's online filing system. The challenge we face here is sourcing the volatility labels for the data. This requires the daily prices for every company, including the ones that went bust or were taken over. Leaving out these firms would introduce significant survivorship bias, as we would automatically be left with safer companies.

The simplest strategy is to take the text of the 10-K reports from EDGAR and compute the volatility ourselves from Yahoo Finance prices, downloaded with `yfinance`, an unofficial Python library. To test the feasibility of this strategy, we used the corpus of \citet{kogan2009}, a published dataset of 10-K reports from 1996 to 2006 that already comes with volatility labels. We applied our strategy to its reports filed in 2004 and compared the volatility we computed from Yahoo with Kogan's label for the same report, looking up each company's ticker in the SEC's list of tickers. Where prices exist, the calculations hold up - our values correlate 0.99 with Kogan's. However, only 26% of the reports (917 of 3,558) can be labelled at all, because the SEC's list only holds tickers for companies that are listed today, and the companies we lose are about 23% more volatile on average. This is exactly the survivorship bias described above. This rules out Yahoo as our source of labels \supp{02-Data, §1}.

To combat this, we instead relied on Kogan's labels themselves. For each report the corpus gives the log volatility over the year before filing, which is our baseline input, and over the year after, which is our target. These labels were computed from CRSP (the Center for Research in Security Prices), the standard academic price database, which keeps a company's prices after it is delisted. 

We now move on to selecting what items to analyse from the reports themselves. Referring to fig ____, we chose to work with only Items 1 and 7. Whilst including the entire reports in our analysis is a potential area for further exploration (Section 5), we elect to work with just these as they show the most promise and every extra section costs more time to extract the text.

** Note: item 1 defo already priced in, hence limited improvment

*Table 1: The main items of a 10-K.*

| Item | Contents | Used | Reason |
|---|---|---|---|
| 1 Business | What the firm does: products, customers, competitors | Yes | Describes products and competitors well enough to define a firm's industry (Hoberg & Phillips, 2016) |
| 7 MD&A | Management's review of the year and its outlook | Yes | Used by Kogan et al.; looks forward as well as back; the most studied section (Li, 2010; Brown & Tucker, 2011) |
| 1A Risk Factors | Management's own list of its main risks | No | Informative (Kravet & Muslu, 2013), but mandatory only from late 2005 (SEC, 2005) |
| 7A Market Risk | Exposure to interest rates, currencies, commodities | No | Mostly sensitivity tables; often already inside Kogan's Item 7 |
| 3 Legal Proceedings | Lawsuits | No | Usually short or "nothing material"; changes over time may matter (Cohen et al., 2020) |
| 8 Financial Statements | The accounts and their notes | No | Numerical, not the language we set out to study |
| 9A, 10–14 | Internal controls; directors, pay, ownership | No | 9A exists only from 2003; Part III usually just refers to the proxy statement |
| 2, 4, 5, 6 | Properties, votes, listing, selected financial data | No | Short, and mostly tables or boilerplate |

Kogan's corpus contains only Item 7 of each report, so we take the rest of the text from EDGAR-CORPUS \citep{loukas2021}, a dataset of 10-K sections extracted from EDGAR. From it we take Item 1, and a second copy of Item 7 that we use to check Kogan's extraction (Section 2.2). This approach also lets us use Kogan's published scores as a benchmark. We also considered Bloomberg, which is available only in person and whose data cannot be shared publicly; WRDS, which provides CRSP but at Bristol is restricted to Business and Economics master's students; and an extension of Kogan's corpus to 2013 by researchers at NCCU, which we could not download \supp{02-Data, §1}.

2.2:
Kogan's reports were matched to EDGAR-CORPUS on the company's SEC number (CIK) and the filing year. Kogan's company name appeared in the EDGAR text for 98.9% of matched reports, with the other cases caused primarily by spelling differences (e.g. "Applebee's" against "APPLEBEES"). We found 95 cases where a company filed two 10-K reports in one calendar year. Since EDGAR-CORPUS identifies a report only by company and year, we could not tell which of the two filings its text belonged to, so we removed both copies of these duplicates. EDGAR-CORPUS is about 14 GB, so we streamed the data and kept only Items 1 and 7 of the reports for which we were able to confirm a match with Kogan. This brought the size of the dataset down to a few hundred MB, which was much more manageable with our available compute. There were also 1,989 cases where EDGAR-CORPUS had no 10-K at all for that company and year, along with 1,144 cases where we were able to identify a match but Item 1 was too short to be useful (fewer than 100 words). In both cases, we cut these data points.

This left us with 27,142 of the original 30,465 (89%) Kogan reports, with the drop in any single year never exceeding 13%. Given the near miss with survivorship bias caused by Yahoo, we again investigated whether the dropped reports had a different risk profile. We ran a Welch t-test on next year's log volatility $y$. Writing $\mu_{\text{kept}}$ and $\mu_{\text{lost}}$ for the mean of $y$ over the kept and lost reports, we tested

$$
H_0:\ \mu_{\text{kept}} = \mu_{\text{lost}}
\qquad \text{against} \qquad
H_1:\ \mu_{\text{kept}} \neq \mu_{\text{lost}}.
$$

Unlike Student's t-test, Welch's test \citep{welch1947} does not assume the two groups have the same variance, which suits groups as unequal in size as ours. We ran the test once for the reports lost because EDGAR-CORPUS had no match, and once for all lost reports. Since $y$ is a logarithm, $100\left(e^{\bar{y}_{\text{lost}} - \bar{y}_{\text{kept}}} - 1\right)$, where $\bar{y}$ is each group's sample mean, gives how much more volatile the lost reports were as a percentage.

| Reports lost | $n_{\text{kept}}$ | $n_{\text{lost}}$ | Lost reports more volatile by | $p$ |
|---|---|---|---|---|
| No EDGAR-CORPUS match | 28,286 | 2,179 | 3.6% | 0.09 |
| For any reason | 27,142 | 3,323 | −1.8% | 0.25 |

Neither difference was significant at the 5% level, so we kept 89% of the corpus without introducing a statistically significant bias.

The one thing that remained was to choose which source to use for Item 7 of the report. Recall that we had extracts from both Kogan and EDGAR and that these were extracted with different software, Kogan's header-matching script and EDGAR-CORPUS's parser. We found that the two sources mostly agreed: the median of the length ratio (EDGAR/Kogan) was 0.94 and 88% of reports fell within 0.5–2×. The exceptions were that EDGAR was much shorter for 9.2% and much longer for 2.4%. We investigated some examples of both cases. We identified the following:

*Table 3: How the two extractions of Item 7 go wrong.*

| What goes wrong | Example | Which source is right |
|---|---|---|
| Kogan stops early at a cross-reference ("...as discussed in note # of") | Newmont Gold (1997), Woodward Governor (2006) | EDGAR-CORPUS |
| EDGAR-CORPUS runs past the end of Item 7 into the exhibit list | Blockbuster (2002) | Kogan |
| The company leaves Item 7 to its *Annual Report to Shareholders* ("incorporated by reference"), so EDGAR-CORPUS's section is only a few lines long | Northeast Utilities (2005) | Kogan, which found the text in the exhibit |

Thus we decided on the following strategy:
1. FinBERT read EDGAR's Item 7 by default because it is clean text with headings (like FinBERT's training text) and comes from the same parser as Item 1.
2. We fell back to Kogan's copy when EDGAR's had under 100 words, which occurred in 2.2% of reports.
3. We kept EDGAR's copy even when it ran past the end of Item 7, since this affected at most 2.4% of reports and added extra text rather than losing any.


2.3:

Since every one of our models is trained to predict Kogan's labels, it was important to perform a few quick sense checks. Recall from the Introduction that the label $y$ is the log of the standard deviation of daily log returns over the 12 months after filing, with the same quantity over the 12 months before filing used as our baseline input. We first checked that the labels were complete and consistent. Every report had a before label and an after label. The labels in our final dataset matched Kogan's raw files for 100% of reports, and there were no missing or infinite values.

Next we checked that the labels looked sensible (Figure 3). Figure 3(a) plots each report's volatility after filing against its volatility before. The points lie close to the diagonal, with a correlation of 0.83, which is exactly why last year's volatility is such a strong baseline and why beating it is hard. Figure 3(b) compares the companies with the market as a whole. Here, and in Table 4 below, we quote volatility per year rather than per day, as is standard in finance. Since daily returns are close to independent, their variances add up over the year, so the annual volatility is

$$
\sigma_{\text{annual}} = \sqrt{252}\,\sigma \approx 15.9\,\sigma.
$$

The median company had a volatility of 50% a year, against 17% for the S&P 500 over the same 12 months, so a typical company was about three times as volatile as the index. This is what we would expect, since a single firm's own news does not average out in the way it does across 500 companies (Campbell et al., 2001).

Both tails of Figure 3(b) stretch a long way, so we looked at the extreme reports individually. We defined these as

$$
\text{very low: } y < -5, \qquad \text{very high: } y > -1.5,
$$

which correspond to a daily volatility $\sigma$ below $e^{-5} \approx 0.7\%$ (calmer than the S&P 500, whose median is about 1.1% a day) and above $e^{-1.5} \approx 22\%$, where a single day's move of 20% was typical. In annual terms, these thresholds are about 11% and 350% a year. We searched each report's text for takeover wording (e.g. "agreement and plan of merger", "tender offer") and distress wording (e.g. "going concern", "Chapter 11"), and checked whether the company filed another 10-K the following year (Table 4).

*Table 4: The extreme labels. Volatilities are annualised medians; the last three columns are the share of reports in each group.*

| Group | Reports | Volatility before | Volatility after | Takeover wording | Distress wording | Filed next year |
|---|---|---|---|---|---|---|
| Very low | 70 | 39% | 6% | 69% | 29% | 4% |
| Ordinary | 27,034 | 53% | 50% | 17% | 31% | 78% |
| Very high | 38 | 158% | 424% | 24% | 82% | 13% |

The very low reports turned out to be takeovers. Once a company has agreed to be bought for a fixed cash price, its shares trade just below that price until the deal closes, so its measured volatility collapses. 69% of these reports contain takeover wording, compared with 17% of ordinary reports, and only 4% of the companies filed again the following year. Most describe the company's own sale, for example DEKALB to Monsanto and Odwalla to Coca-Cola. Importantly, these companies were not risk-free; they were about to be delisted. The very high reports were the opposite case of companies being in distress. 82% contain distress wording, such as a going-concern warning from the auditors or a Chapter 11 filing (e.g. Salon Media and eMagin). Unlike the takeovers, these companies were already very volatile the year before (a median of 158%).

Since these are genuine prices rather than data errors, we kept them. However, they matter far more than their numbers suggest. The 70 very low reports make up only 0.26% of the data but carry 10% of the baseline's squared error, and removing them alone would cut the baseline MSE from 0.128 to 0.116. They are also where the text has the clearest chance of beating the baseline, since a report announcing "an agreement and plan of merger" says something about next year that we would not otherwise have known from just last year's volatility.


2.4

The last step was transform our large blocks of text into something a model can learn from. For this we used FinBERT \citep{araci2019}, a version of BERT \citep{devlin2019} trained further on financial text. We did not change FinBERT itself; we only used it to read the reports, and the models in Section 3 were trained on what it produced. This is a good use case for transfer learning: FinBERT has already learnt from far more text than our 27,142 reports, so we reuse what it knows rather than learning the language from scratch.

FinBERT can only read 512 tokens at a time, where a token is a word or a piece of a word, and our reports are much longer than this. So for each report we treated Item 1 and Item 7 separately and did the following:

1. Split the text into FinBERT's own tokens.
2. Cut the tokens into consecutive chunks of 510, since two of the 512 slots are taken by FinBERT's start and end markers.
3. Kept at most 64 chunks per section, recording the true number.
4. Passed each chunk through FinBERT and kept its CLS vector, the 768 numbers FinBERT uses to summarise the whole chunk.

*Table 5: Chunks per report.*

| Section | Median chunks | 90th percentile | Reports cut at 64 | Chunks stored |
|---|---|---|---|---|
| Item 1 | 14 | 31 | 0.27% | 444,394 |
| Item 7 | 16 | 33 | 0.57% | 498,607 |

A typical report gave 32 chunks in total, and fewer than 1% of reports were cut at 64 chunks, so the cap lost very little text. We saved the vectors at half precision (float16), which halves the storage to about 1.4 GB for all 943,001 chunks. Reading every report took FinBERT about 12 hours on a laptop GPU, so we did this once and every model then reused the saved vectors \supp{02-Data, \S6}. The ridge regression averages a report's chunks into a single vector, while the attention network keeps them separate and learns how much weight to give each one (Section 3).

The reports also changed over the decade. The median length of item 7 roughly doubled in this time, with the steepest rise around the Sarbanes–Oxley Act of 2002, which tightened reporting standards, and \citet{kogan2009} note the same jump. 

This matters for how we split the data into training and test sets. Volatility moves with the market as a whole - it was high around the 2000–2002 crash and low by 2005. If we split the reports at random, a model could learn each year's market sentiment from that year's own reports, and look better than it really is. So, following \citet{kogan2009}, we split by year. Every model was trained only on earlier years and tested on one year at a time, from 2001 to 2006, apart from the fine-tuned FinBERT, which was tested on 2006 only. This also lets us compare our scores directly with the paper's.

However, there is one subtlety. A report filed in December 2005 has a label covering December 2005 to December 2006, which overlaps the 2006 test year. Training on it leaks a little information about 2006's market into the model. We therefore used two training windows:

- Paper: the five years before the test year, as in \citet{kogan2009}, for direct comparison.
- Strict: the four years ending two years before the test year, so that no training label overlaps the test period.

*Table 6: Training and test years. Train/test give the number of reports.*

| Test year | Paper window | Train | Strict window | Train | Test |
|---|---|---|---|---|---|
| 2001 | 1996–2000 | 9,747 | 1996–1999 | 7,596 | 2,299 |
| 2002 | 1997–2001 | 10,788 | 1997–2000 | 8,489 | 2,587 |
| 2003 | 1998–2002 | 11,399 | 1998–2001 | 8,812 | 3,217 |
| 2004 | 1999–2003 | 12,474 | 1999–2002 | 9,257 | 3,168 |
| 2005 | 2000–2004 | 13,422 | 2000–2003 | 10,254 | 3,121 |
| 2006 | 2001–2005 | 14,392 | 2001–2004 | 11,271 | 3,003 |

The strict window trains on about a fifth fewer reports, and the training years end further from the test year, so we expect every model to score a little worse on it. If a model's gain over the baseline holds up on the strict window, we can be more confident it comes from the text rather than from leaked information about the test year's market. Within each window, we tuned hyperparameters on the last training year and never on the test year. Every model read its splits from the same file, so all of them were scored on exactly the same reports \supp{02-Data, \S7}.


5: **[FIRST DRAFT WRITTEN BY CLAUDE — to edit]** Computational cost and scaling (proposed new name for Section 5)

Almost all of our computation went into one step, FinBERT reading the text. FinBERT is a 110-million-parameter network (the BERT-base architecture of \citet{devlin2019}), and it had to read all 943,001 chunks. The attention network, on the other hand, only works on the saved 768-number summaries, and one epoch of it took about six seconds on a Colab T4 GPU. So, when we ask how our approach scales, we are really asking how the reading step scales. We investigated this by timing FinBERT on a laptop's CPU, its GPU (an Apple M3) and a Colab T4, at different precisions and batch sizes \supp{04-Scaling}.

*Table 7: FinBERT's speed in chunks per second. The T4 was timed on Yang et al.'s FinBERT, which has the same architecture as ours and so does the same work per chunk.*

| Machine | Precision | Speed test | Full run |
|---|---|---|---|
| Laptop CPU | fp32 | 10 | – |
| Laptop GPU (M3) | fp32 | 22 | – |
| Laptop GPU (M3) | fp16 | 26 | 22 |
| Colab T4 | fp16 | 158 | 118 |

The T4 was about 6 times faster than the laptop's GPU, which in turn was only 2–3 times faster than its CPU. Using half precision (fp16) instead of 32-bit numbers sped the laptop up by about 19% and halved the size of the saved output, since 16-bit numbers take half the memory and GPUs can process them faster \citep{micikevicius2018}. A check on 50 reports found no measurable change to the vectors. We also found that batch size barely mattered on either GPU. One 510-token chunk is already enough work to keep the GPUs busy, so feeding them more chunks at once adds nothing. We also timed FinBERT on 32 up to 1,024 chunks and found that the time grows linearly with the number of chunks ($R^2 = 0.997$), with no measurable start-up cost. FinBERT therefore costs a fixed price per chunk.

Both full runs were slower than their speed tests, for different reasons. The laptop ran at its speed-test rate for the first hour and a half, then dropped by about 15% and stayed there for the remaining ten hours. This is likely due to thermal throttling, although we did not record the temperature to confirm it. The T4 ran at a steady rate throughout, so heat was not the issue. Instead, for each batch of reports, Colab first read the text and split it into tokens on its CPU, and only then sent it to the GPU, so the GPU sat idle while the CPU worked. If the GPU alone reads $g$ chunks per second and the CPU step prepares $p$, then done one after the other the rate is

$$
\text{rate} = \frac{1}{1/g + 1/p}.
$$

With $g = 158$ and the measured rate of 118, this gives $p \approx 460$ chunks per second. This matters for scaling, because the CPU step does not get faster when the GPU does, so the faster the GPU, the larger the share of time it spends waiting. This is Amdahl's law \citep{amdahl1967,dst091}. There are two simple fixes. The first is to run the CPU step alongside the GPU, as PyTorch's `DataLoader` does, which raises the rate to $\min(g, p)$. The second is to tokenise the text once in advance and store the token ids instead of the raw text.

Next, we looked at how far our data could realistically grow. Our corpus is only a part of what exists. Counting the SEC's own index of every filing, 249,368 10-Ks were filed from 1996 to 2025, and in 2002–2006 Kogan's corpus covered only about a third of them. This is because it only keeps companies whose shares CRSP prices and whose Item 7 Kogan's script could find. We also measured how long recent reports are, using a sample of EDGAR-CORPUS \citep{loukas2021} for 2006, 2013 and 2020. Items 1 and 7 barely grew, from a mean of about 11,600 words to 13,600, while the whole 10-K grew from about 30,800 words to 43,900, mostly because the Risk Factors section (Item 1A) tripled in length. Since FinBERT costs a fixed price per chunk, the cost of each larger dataset follows directly from its number of chunks (Table 8) \supp{04-Scaling, \S6}.

*Table 8: Realistic ways to grow the data, including a 1000× increase as suggested in the brief. Times are for one Colab T4 at the rate of our full run (118 chunks per second).*

| Data | Reports | Times our data | One T4 | FinBERT output |
|---|---|---|---|---|
| Our corpus (Kogan's companies, 1996–2006) | 27,142 | 1 | 2 hours | 1.4 GB |
| Our companies, extended to 2025 | 78,635 | 3.3 | 7 hours | 4.8 GB |
| Every 10-K filer, 1996–2025 | 249,368 | 10 | 22 hours | 15 GB |
| Every 10-K filer, whole 10-K | 249,368 | 30 | 2.7 days | 43 GB |
| The brief's 1000 times | 27 million | 1000 | 93 days | 1.45 TB |

Extending our companies up to today, adding every company that files a 10-K, and even reading every section of every report all run on a single GPU in under three days. So for any realistic version of this study, compute is not the bottleneck. The real limit is the labels. Every report needs a volatility label, which means daily prices for every company, including the ones that were later delisted. Kogan's labels stop in 2006, and the only source that keeps delisted companies is CRSP, which at Bristol is only available through WRDS to Business and Economics master's students. As we saw in Section 2.1, Yahoo Finance would have labelled only 26% of the reports and demonstrated considerable survivorship bias. Not every filer has traded shares either (for example funds, trusts and companies that have only issued debt), so "every 10-K filer" is an upper bound on what could ever be labelled. With access to CRSP though, repeating our study on every company up to today using the same two items would take about a day on one GPU (22 hours in Table 8).

The brief also asks how our approach would cope with 1000 times more data. This is about 27 million reports, far more 10-Ks than have ever been filed, so it would mean adding other text, such as quarterly 10-Q reports, 8-K filings, earnings-call transcripts and news. Since the cost is linear, this would take about 93 days on one T4, and produce about 1.45 TB of FinBERT output. However, each chunk is read independently of every other, so the work splits cleanly across GPUs. In MapReduce terms \citep{dst092}, reading is the map, and the only reduce is collecting each report's vectors back together, which costs almost nothing. With $N$ GPUs,

$$
\text{time}(N) = \text{start up} + \frac{\text{input size}}{\text{bandwidth}} + \frac{\text{chunks}}{N \times \text{rate per GPU}},
$$

where the first two terms do not shrink as GPUs are added. Assuming the input comes from one shared store at 1 GB/s, 1,024 T4s, each running at its speed-test rate of 158 chunks per second, would read 1000 times our data in about 2.3 hours, at 71% of perfect efficiency, or 1.9 hours and 86% if they read pre-tokenised ids. Beyond this point the shared store becomes the limit, and the fix is to split the data as well as the work, with each worker reading its own shard from its own disk, as Spark or MapReduce do \citep{dean2004}.

Training the models that use the saved vectors would not need a cluster. Even at 1000 times the data, one epoch of the attention network would take about 1.6 hours on a single T4. The ridge regression is cheaper still, since it only needs the sums $X^\top X$ and $X^\top y$, which can be computed separately for each year (or each shard of reports) and then added together, another map with a simple reduce. The word count models scale in a similar way. Their word count matrix takes 0.19 GB stored sparsely, which would be about 190 GB at 1000 times the data. This is too big to hold in memory, but it is easy to split into shards. Counting the words took 91 seconds on one CPU core, which would be about 25 hours at 1000 times, or under half an hour on 64 cores.

Fine-tuning is the exception. Here FinBERT itself is trained, so every chunk has to go through the network forwards and backwards on every step. On a T4 this ran at about 40 chunks per second, a quarter of the speed of only reading the text. Even reading just 4 random chunks of each report per step, a single test year with a single seed took about 1.4 GPU-hours, against about 15 minutes for all 60 runs of the attention network on Yang's saved vectors. At 1000 times the data, one epoch would be about 45 million training chunks, or about 310 T4-hours. Unlike reading, the training steps are not independent, since every step updates the same 110 million weights. The work can still be split by giving each GPU part of every batch (data parallelism), but after every step the gradients have to be added up across all the GPUs and sent back. This exchange takes longer the more GPUs there are \citep{dst092}, so each extra GPU speeds up fine-tuning by less than it would speed up reading. Since in our one test it did not clearly beat the models on frozen vectors (Section 3), it is hard to justify at this scale.

These estimates have clear limits. We tested only one laptop and one type of cloud GPU, and the T4 figures come from Yang's FinBERT rather than ours. The storage bandwidth is assumed rather than measured, and fine-tuning was timed on one test year and one seed. The growth after 2006 assumes we keep the same share of filers, and is based on a sample of just three years. Finally, the larger figures are extrapolations, since we only measured the cost per chunk up to about a million chunks.
