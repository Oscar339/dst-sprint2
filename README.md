# Data Science Toolbox — Sprint 2: Data at Scale (Neural Networks)

Assessed Sprint 2 for [Data Science Toolbox](https://dsbristol.github.io/dst/)
(MATHM0029, University of Bristol, 2026-27). Brief:
[Assessment2.pdf](https://dsbristol.github.io/dst/assets/assessments/Assessment2.pdf).
Laid out after the unit's
[example project](https://github.com/dsbristol/dst_example_project), as were
[Sprint 0](https://github.com/Oscar339/dst-sprint0) and
[Sprint 1](https://github.com/Oscar339/dst-sprint1).

**Deadline: Wednesday 2 December 2026, 12:00 (week 11).**
Group content is final by **Monday 30 November, 12:00** (48 hours before), so
each of us can write the individual reflection.

## The task, in one paragraph

Choose an application domain and apply **neural networks** in a way that would
scale to a deployable data-science solution: (a) pick a scientific / analysis
question, (b) find out how the model performs computationally, and (c) apply it
to answer the question. It must run on a laptop or Colab, but we must argue how
it would work if the data grew by a **factor of 1000**, and that it would make
efficient use of high-performance resources (GPUs, clusters) at that scale.

Credit goes to the implementation (beyond off-the-shelf, multiple
implementations, the maths of our choices), the application (a well-chosen and
well-documented dataset, compared with a "real" data-science setting), the
correctness of the methods, and the robustness of the conclusions.

## Project Group

* Oscar Butler — _to be agreed_
* Alex Robinson — _to be agreed_
* Louis Mulryan — _to be agreed_

Equity: _to be agreed before the work gets underway_ (the brief asks for this).

## Domain and data

_To be chosen._ Document the search for data as we go in
[`data/README.md`](data/README.md): every candidate considered and why it was
kept or rejected, since the brief rewards showing there is no more appropriate
source.

## Reading order

All report content is in `report/`, numbered `NN-Name.ext` in reading order.
_To be filled in as the report is written._

1. `01-Data.ipynb` — getting the data, exploring it, and any shared split.
   **Run this first**: later notebooks load its output.

### Requirements

Python 3.13. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Neural networks use [PyTorch](https://pytorch.org/). It runs on CPU; it uses a
GPU (CUDA, or Apple `mps`) automatically where the notebook supports it.

Data is downloaded by `report/01-Data.ipynb` into `data/raw/` (not committed).
Any manual step is stated in the notebook where it happens.

## Evidence

Each member's working, including dead ends, is in their own folder:

* `Oscar/`
* `Alex/`
* `Louis/`

## References

### Data

_To be added once the dataset is chosen._

### Papers

_Add as we go, and link our results to those on analogous problems. Websites
and other non-paper sources count too: cite them._
