# tti-explorer
![Python Tests](https://github.com/rs-delve/tti-explorer/workflows/Python%20Tests/badge.svg)
[![codecov](https://codecov.io/gh/rs-delve/tti-explorer/branch/master/graph/badge.svg)](https://codecov.io/gh/rs-delve/tti-explorer)

This is a repository for `tti-explorer`, a library for simulating infection spread. This library is built to explore the impact of various test-trace-isolate strategies and social distancing measures on the spread of COVID-19 in the UK.

Our model builds upon the model of [Kucharski et al. (2020)](https://www.medrxiv.org/content/10.1101/2020.04.23.20077024v1). We use the BBC Pandemic dataset ([Klepac et al. (2018)](https://researchonline.lshtm.ac.uk/id/eprint/4647173/)).

This work was conducted for the Test, Trace, Isolate project as part of the Royal Society's DELVE initiative. This repository facilitiates simulations for a technical report which can be found [here](https://rs-delve.github.io/pdfs/2020-05-27-effectiveness-and-resource-requirements-of-tti-strategies.pdf). These simulations inform the report for the Test, Trace, Isolate project which can be found [here](https://rs-delve.github.io/reports/2020/05/27/test-trace-isolate.html). Exact version of this repo used in the report is available as a [tag](https://github.com/rs-delve/tti-explorer/releases/tag/rs-delve-tech-report-0520).

## Introductory notebook: 

For an introduction to the functionality offered by `tti-explorer`, start with the [tti-experiment notebook](https://colab.research.google.com/github/rs-delve/tti-explorer/blob/master/notebooks/tti-experiment.ipynb).

## Requirements:
### tti_explorer
- Python 3.6+
- numpy
- scipy
- pandas
- matplotlib
- dataclasses (for Python 3.6)
### scripts, tests and notebooks
- jupyter
- tqdm
- pytest


## Setup:
```bash
git clone https://github.com/rs-delve/tti-explorer
cd tti-explorer
pip install -r requirements.txt
pip install .
```
