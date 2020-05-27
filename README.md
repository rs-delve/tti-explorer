# tti-explorer
This is a repository for `tti-explorer`, a library for simulating infection spread. This library is built to explore the impact of various test-trace-isolate strategies and social distancing measures on the spread of COVID-19 in the UK.

Our model builds upon the model of [Kucharski et al. (2020)](https://www.medrxiv.org/content/10.1101/2020.04.23.20077024v1). We use the BBC Pandemic dataset ([Klepac et al. (2018)](https://researchonline.lshtm.ac.uk/id/eprint/4647173/)). 

## Introductory notebook: 

For an introduction to the functionality offered by `tti-explorer`, start with the [tti-experiment notebook](https://colab.research.google.com/github/rs-delve/tti-explorer/blob/master/notebooks/tti-experiment.ipynb).

## Requirements:
### tti_explorer
- Python >= 3.8
- numpy
- scipy
- pandas
- matplotlib
### scripts, tests and notebooks
- jupyter
- tqdm
- pytest


## Setup:
```bash
    git clone https://github.com/rs-delve/tti-explorer
    cd tti-explorer
    pip install .
```
