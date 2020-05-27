# tti-explorer
This is a repository for `tti-explorer`, a library for simulating infection spread. This library is built to explore the impact of various test-trace-isolate strategies and social distancing measures on the spread of COVID-19 in the UK.

## Introductory notebook: 

For an introduction to the functionality offered by `tti-explorer`, start with the [tti-experiment notebook](https://github.com/rs-delve/tti-explorer/blob/master/notebooks/tti-experiment.ipynb).

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
    python setup.py sdist bdist_wheel
    pip install .
```
