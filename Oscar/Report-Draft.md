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
Here $y$ is called the log volatility. We use both the log volatility and the log returns as these often play nicer than their standard equivalents. For example, one useful property is the volatilities of companies are right-skewed, applying a log to volatility turns this into a more familiar normal-ish distribution \citep{andersen2001}. They also carry the benefit of allowing us to compare our results directly to the paper this report is inspired by \citep{kogan2009}.

As a rule of thumb, the volatility of a company's returns are relatively consistent each year. This is because 'risky' stocks tend to have idiosyncratic properties that persist over time that make them more prone to risk than other stocks. For example the volatility of NVIDIA will almost always be higher than that of PepsiCo. This is because people will always drink Pepsi but AI infrastructure demand remains a hot topic of debate.

With this in mind, predicting the next year's volatility for a company based on just the volatility of previous years is a simple and effective method. We will use this as the baseline and investigate how adding information about a company's SEC filings changes prediction accuracy. Following Kogan et al. (2009), we predict the log of volatility and score models by their mean squared error (MSE).

We use about 27,000 10-K reports from 1996 to 2006. We start by building a ridge regression, the simplist model we can build that looks at the text. Then we build an attention network and an Multi Layer Perceptron (MLP) in an attempt to beat this baseline. We also experiment with different encoders to determine the effect of different tokenisation techiniques on text understanding.

Its worth noting hre that even a negative result is informative. If we find that including the text of the 10-K filings does not improve prediction accuracy then this implies something potentially more interesting. It would suggest that the information in the report is already reflected in how the share price has moved, and so is already contained in last year's volatility. This is what one would expect in an efficient market, where the price of a stock should reflect all publicly available information. This is a theme that we will return to throughout the report as we discover many of our models do not yield significant improvements over the baseline. **[TODO: add numbers once models are trained]**

data:

When sourcing data for the project, we need reliable sources for both the 10-K reports and the share price volatility in the year after they were filed. The 10-K reports are easy enough to source; every report is public on EDGAR, the SEC's online filing system. The challenge we face here is sourcing the volatility labels for the data. This requires the daily prices for every company, including the ones that went bust or were taken over. Leaving out these firms would introduce significant survivorship bias, as we would automatically be left with safer companies.

The simplest strategy is to take the text of the 10-K reports from EDGAR and compute the volatility ourselves from Yahoo Finance prices, downloaded with `yfinance`, an unofficial Python library. To test the feasibility of this strategy, we apply it to the reports in Kogan's corpus filed in 2004, for which the correct labels are already known. We look up each company's ticker in the SEC's list of tickers and compare the volatility we compute from Yahoo with Kogan's label for the same report. Where prices exist, the calculations hold up - our values correlate 0.99 with Kogan's. However, only 26% of the reports (917 of 3,558) can be labelled at all, because the SEC's list only holds tickers for companies that are listed today, and the companies we lose are about 23% more volatile on average. This is exactly the survivorship bias described above. This rules out Yahoo as our source of labels \supp{02-Data, §1}.

To combat this, we instead rely on the 10-K corpus of \citet{kogan2009}, which covers reports filed from 1996 to 2006. For each report it gives the log volatility over the year before filing, which is our baseline input, and over the year after, which is our target. These labels were computed from CRSP (the Center for Research in Security Prices), the standard academic price database, which keeps a company's prices after it is delisted. 

We now move on to selecting what items to analyse from the reports themselves. Refereing to fig ____ we chose to work with only items 1 and 7. Whilst including the entire reports in our analysis is a potential area for further exploration [potential link to later section on scalability] we elect to work with just these as they show the most promise and every extra section costs more time to extract the text.

** Note: item 1 defo already priced in, hence limited improvment

*Table 1: The main items of a 10-K.*

| Item | Contents | Used | Reason |
|---|---|---|---|
| 1 Business | What the firm does: products, customers, competitors | Yes | Industry and business model drive much of a firm's risk (Hoberg & Phillips, 2016) |
| 7 MD&A | Management's review of the year and its outlook | Yes | Used by Kogan et al.; looks forward as well as back; the most studied section (Li, 2010; Brown & Tucker, 2011) |
| 1A Risk Factors | Management's own list of its main risks | No | Informative (Kravet & Muslu, 2013), but mandatory only from late 2005 (SEC, 2005) |
| 7A Market Risk | Exposure to interest rates, currencies, commodities | No | Mostly sensitivity tables; often already inside Kogan's Item 7 |
| 3 Legal Proceedings | Lawsuits | No | Usually short or "nothing material"; changes over time may matter (Cohen et al., 2020) |
| 8 Financial Statements | The accounts and their notes | No | Numerical, not the language we set out to study |
| 9A, 10–14 | Internal controls; directors, pay, ownership | No | 9A exists only from 2003; Part III usually just refers to the proxy statement |
| 2, 4, 5, 6 | Properties, votes, listing, selected financial data | No | Short, and mostly tables or boilerplate |

Kogan's corpus contains only Item 7 of each report, so we take the rest of the text from EDGAR-CORPUS \citep{loukas2021}, a dataset of 10-K sections extracted from EDGAR. From it we take Item 1, and a second copy of Item 7 that we use to check Kogan's extraction (Section 2.2). This approach also lets us use Kogan's published scores as a benchmark. We also considered Bloomberg, which is available only in person and whose data cannot be shared publicly; WRDS, which provides CRSP but at Bristol is restricted to Business and Economics masters students; and an extension of Kogan's corpus to 2013 by researchers at NCCU, which we could not download \supp{02-Data, §1}.

2.2:
Kogan's reports were matched to EDGAR-CORPUS on the company's SEC number (CIK) and the filing year. Kogan's company name appeared in the EDGAR text for 98.9% of matched reports, with the other cases caused primarily by spelling differences (e.g. "Applebee's" against "APPLEBEES"). We found 95 companies that filed two 10-K reports in one calendar year. Since EDGAR-CORPUS identifies a report only by company and year, we could not tell which of the two filings its text belonged to, so we removed both copies of these duplicates. EDGAR-CORPUS is about 14 GB, so we streamed the data and kept only Items 1 and 7 of the reports for which we were able to confirm a match with Kogan. This brought the size of the dataset down to a few hundred MB, a much more manageable task for our available compute. There were also 1,989 cases where EDGAR-CORPUS had no 10-K at all for that company and year, along with 1,144 cases where we were able to identify a match but Item 1 was too short to be useful (fewer than 100 words). In both cases, we cut these data points.

This left us with 27,142 of the original 30,465 (89%) Kogan reports, with the drop in any single year never exceeding 13%. Given the near miss with survivorship bias caused by Yahoo, we again investigated whether the dropped reports had a different risk profile. We ran a Welch t-test on next year's log volatility $y$. Writing $\mu_{\text{kept}}$ and $\mu_{\text{lost}}$ for the mean of $y$ over the kept and lost reports, we tested

$$
H_0:\ \mu_{\text{kept}} = \mu_{\text{lost}}
\qquad \text{against} \qquad
H_1:\ \mu_{\text{kept}} \neq \mu_{\text{lost}}
$$

using the statistic

$$
t = \frac{\bar{y}_{\text{lost}} - \bar{y}_{\text{kept}}}{\sqrt{\dfrac{s_{\text{kept}}^2}{n_{\text{kept}}} + \dfrac{s_{\text{lost}}^2}{n_{\text{lost}}}}},
\qquad
\nu = \frac{\left(\dfrac{s_{\text{kept}}^2}{n_{\text{kept}}} + \dfrac{s_{\text{lost}}^2}{n_{\text{lost}}}\right)^2}{\dfrac{\left(s_{\text{kept}}^2/n_{\text{kept}}\right)^2}{n_{\text{kept}}-1} + \dfrac{\left(s_{\text{lost}}^2/n_{\text{lost}}\right)^2}{n_{\text{lost}}-1}},
$$

where $\bar{y}$, $s^2$ and $n$ are each group's sample mean, variance and size, and $t$ is compared with a $t$-distribution on $\nu$ degrees of freedom. Unlike Student's t-test, Welch's test does not assume the two groups have the same variance, which suits groups as unequal in size as ours. We ran the test once for the reports lost because EDGAR-CORPUS had no match, and once for all lost reports. Since $y$ is a logarithm, $100\left(e^{\bar{y}_{\text{lost}} - \bar{y}_{\text{kept}}} - 1\right)$ gives how much more volatile the lost reports were as a percentage.

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