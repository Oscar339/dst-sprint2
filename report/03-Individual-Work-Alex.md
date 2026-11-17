# Individual work - Alex

In this project, I built the scoring metrics for the models, including a linear (Ridge) baseline that the neural networks
have to beat to be proven effective. I then built an attention model and tested a second encoder. To see all of my full workings, 
please head to `Alex/` where all my research, planning and modelling is. 

## 1. Scoring

Firstly, I built two things for each neural network to be compared against. As we can score the actual accuracy of 
each model on the given volatility for the test year, the benchmark is going to be against other models; in particular
we want to see whether a neural network can do better than a simple linear model. So, each model is compared with two baselines. 
The first is copying last year's volatility (Kogan et al.'s baseline), and the second is a straight line fitted on it, 
and the difference in MSE gets a paired bootstrap interval over the test reports (as seen in lecture 05.2).

The most useful thing this section does is split each test year's MSE into two parts, $\text{MSE} = \bar e^{2} +
\tfrac1n\sum_i (e_i - \bar e)^2$. We look at the level, how far off the model is from the whole year's volatility, which
no single report can know, and the spread, how well the model ranks the companies within the year, which is what the
text could help with. For the straight line, the level is 44% of the MSE in 2003 and 30% in 2001, so a model can look
much better or worse than it is just by getting the year right or wrong.

## 2. The linear baseline

A ridge regression on each report's averaged FinBERT vectors, fitted to what the straight line gets wrong, is the
simplest text model, and a neural network with no hidden layer, so the networks have to beat it to show their hidden
layers help.

* **Fitting all years together was a dead end.** Ridge swung from 26% better than the line to 16% worse depending
  on the year, and its biggest gain (2003) was almost all level: the text was being used to guess which era a report
  is from. So the final version only lets the text explain differences between companies in the same year.
* **Within a year, the text adds little.** The within-year ridge changes the MSE by −1.0% to +0.3% on average
  against the line, helping in 2001 and 2002 and hurting in 2004. In 11 of 72 fits the last training year chose to
  switch the text off.
* **It is fitted from per-year sums.** Ridge only needs sums such as $X^\top X$ over the training reports, so they
  are computed once per year and added up for any window: the map/reduce of lecture 09.2. Each fit then costs the
  same however many reports there are, and the notebook went from 45 minutes to 3.

Comparing with the paper gave the result I found most interesting. On Kogan et al.'s own full data and windows, the
straight line with no text scores 0.1403 for 2006 against 0.1402 for their best text model, so most of their gain over
copying comes from fitting last year's volatility, not from the text. I also found that the 11% of reports our data
could not match to EDGAR-CORPUS are the hard ones, with 3 to 6 times the error of copying, so our MSEs are lower than
the paper's for reasons that have nothing to do with any model. FETILDA's reported 27% gain over Kogan's baseline is
measured on a different subset of reports, so it may partly be the same effect.

## 3. The attention network

The main model reads each report as a bag of chunk vectors, learns a weight for each chunk with attention pooling
(Ilse et al., 2018) and predicts the within-year residual. The pieces come from the course where they can: a softmax
over chunk scores, ReLU and tanh layers, and the training loop of workshop 06.3 (lectures 06.1 and 06.2). Early
stopping, dropout and AdamW come from the literature. Before training I checked that the weights sum to one, that
padding gets no weight, that shuffling the chunks does not change the prediction, and that the network can overfit
32 reports.

It was trained on a Colab T4: 4 variants (attention on Item 7, Item 1 and both; mean pooling on Item 7) × 5 seeds ×
2 windows × 6 test years, 240 runs in 1.2 GPU-hours, saved to Google Drive so a dropped session could resume.

* **It does no better than the within-year ridge.** On Item 7, both change the MSE by −1.0% on average on the paper
  window, with the same pattern by year.
* **Learning the weights adds nothing measurable.** Attention beats mean pooling by 0.3 percentage points, less than
  the spread between seeds. The weights are close to equal (the effective number of chunks is 78% of the actual
  number), which is why.
* **It prefers Item 7.** The model given both sections puts 78% of its weight on Item 7, which is 62% of the chunks,
  and Item 1 on its own is worse than the line.

Measured against copying last year, as FETILDA measured, the network wins in four of six years by up to 13%, but so
does the straight line with no text, by almost the same amounts. Our comparison separates the two, which the papers'
comparisons do not.

## 4. The second encoder (in progress)

By the decision rule in the plan, the evidence now points at the vectors rather than at how they are combined. Our
vectors come from `ProsusAI/finbert`, which was tuned for sentence sentiment. Notebook 04 reads Item 7 again with Yang
et al.'s FinBERT, pre-trained on 10-Ks and analyst reports, keeping both the CLS and the token-average vector of each
chunk, and reruns the same ridge and attention network on them, changing nothing else. It is built and checked, and
runs on Colab once Oscar shares the Item 7 text.

## 5. Scaling

Three parts of my work bear on the 1000× question. The per-year sums make ridge's cost after one pass independent of
the number of reports. The 240 training runs are independent of each other, so they are embarrassingly parallel
(lecture 09.1); within a run the epochs are sequential, which is the part Amdahl's law says more hardware cannot
remove (lecture 09.2). And embedding every chunk with FinBERT is the expensive step, but every chunk is independent,
so it splits across as many GPUs as are available; notebook 04 times it at five batch sizes.

## 6. What I take from it

Every model I built agrees: once last year's volatility is used properly and the year's level is taken out, the
FinBERT text adds little, and a neural network adds nothing over a linear model on the same vectors. That is a
negative result, but a careful one, and it changes how the published results read. The second encoder tests the one
explanation that is left.

## References

- Ilse, M., Tomczak, J. M. & Welling, M. (2018). Attention-based deep multiple instance learning. *ICML 2018*.
- Kogan, S., Levin, D., Routledge, B. R., Sagi, J. S. & Smith, N. A. (2009). Predicting risk from financial reports with regression. *NAACL-HLT 2009*, 272–280. https://aclanthology.org/N09-1031/
- Xia, B., Rawte, V. D., Zaki, M. J. & Gupta, A. (2022). FETILDA: An effective framework for fin-tuned embeddings for long financial text documents. *arXiv:2206.06952*.
- Yang, Y., Uy, M. C. S. & Huang, A. (2020). FinBERT: A pretrained language model for financial communications. *arXiv:2006.08097*.
- Data Science Toolbox (University of Bristol, 2026): lectures 02.2, 05.2, 06.1, 06.2, 09.1 and 09.2, and workshop 06.3. https://dsbristol.github.io/dst/coursebook.html
