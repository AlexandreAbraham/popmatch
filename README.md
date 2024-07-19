# Popmatch

A population matching package for python.
⚠️ This is a research project, the code is not production ready and not guaranteed to work ⚠️
It proposes the original code of psmpy along with mapping over R's Matchit package.

## Installation

You can install this package using pip:

```bash
pip install .
```

## Running experiments

Experiments are located in the experiments folder. Each folder contains a script called `compare_propensity_matching.py` that will run the experiments as presented in our paper. The top script `run.sh` will crawl each directory and launch the script in sequence. Once each experiment has run, all results are cached and stored in their respective directories. In order to obtain the results table as presented in the paper, three scripts in the top level folder will browse the experiment folder, gather the information needed, and format it. Those scripts are `gather_correlation.py`, `gather_overlap.py`, and `gather_results.py`.
