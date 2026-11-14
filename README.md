# FPF: fuzzy prototype federated learning for few-shot bearing fault diagnosis

Code that reproduces every table, figure and number of the manuscript

> K. Kaushik, A. S. Mahajan, D. P. Uddandarao, M. Altayeb, G. K. Rajput and A. Alkhayyat,
> *Fuzzy prototype federated learning for few-shot bearing fault diagnosis under heterogeneous operating conditions and label skew*,
> submitted to Applied Soft Computing (special issue on fault diagnosis and health management soft computing methodologies for Industrial 5.0 systems).

The repository contains the feature extractor, the federated training code (FPF, the ablations, eight baselines and a centralised reference), the differential-evolution tuner, all experiment drivers, the analysis code that writes the paper's tables, figures and number macros, and a verification script.

## Layout

```
fpf/
  config.py        constants, default hyperparameters, paths
  features.py      46 sensor-agnostic features per vibration window
  data.py          dataset loaders, windowing, feature cache, time-based protocol
  model.py         encoder, fuzzy prototype loss, local training, typicality weights
  federated.py     client construction (operating-condition clients, cross-rig clients), federated loops, baselines
  de_search.py     differential evolution over (m, mu, lambda, gamma) with a leave-one-client-out transfer score
  experiments.py   experiment drivers (main, ablation, new client, corrupted client, sensitivity, convergence, ...)
  analysis.py      statistics, tables, figures and numbers.tex
  signals.py       example-signal figure
  graphical.py     graphical abstract (bar heights read from the results)
  verify.py        consistency checks between stored results and a fresh recomputation
scripts/
  download_data.sh fetches the three public dataset repositories at pinned commits
  run_de.sh        runs the six DE searches
  run_all.sh       the complete pipeline
results/           raw result tables (CSV/pickle/JSON) produced by the run that is reported in the paper
paper_assets/      tables/*.tex, figs/*.pdf and numbers.tex generated from results/
tests/             unit and protocol tests
paper/             LaTeX source of the manuscript with its tables, figures and number macros
```

## Installation

Python 3.12 or newer is recommended. The pinned versions are those used for the reported run.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

```bash
./scripts/download_data.sh data
export FPF_DATA_DIR=$PWD/data
```

The script clones the following public repositories at fixed commits (sparse checkout for the first):

| Dataset | Repository | Commit |
|---|---|---|
| CWRU (fan end) and SEU gearbox | `cathysiyu/Mechanical-datasets` | `9732a28` |
| JNU | `ClarkGableWang/JNU-Bearing-Dataset` | `75b3361` |
| HIT-SM (HIT-A: SpectraQuest MFS, HIT-B: self-built rig) | `hitwzc/Bearing-datasets` | `12e46cd` |

The data remain under the terms of their original providers; please cite them as in the paper.

## Reproducing the paper

```bash
./scripts/run_all.sh        # about one hour on two CPU cores
```

or step by step:

```bash
python -m fpf.data                                   # feature cache (about one minute)
./scripts/run_de.sh                                  # six DE searches -> results/de_n{1,3,5}_s{0,1}.json
python -m fpf.experiments main --skew 0              # balanced label sets
python -m fpf.experiments main --skew 1              # label-skewed clients
python -m fpf.experiments abl_default --skew 0
python -m fpf.experiments abl_default --skew 1
python -m fpf.experiments newclient robust crossrig confusion emb sens conv
python -m fpf.signals
python -m fpf.analysis                               # tables, figures, numbers.tex
python -m fpf.graphical
python -m fpf.verify                                 # checks that stored results are reproduced
```

The manuscript reads every quoted number through `\nv{key}` macros defined in `paper_assets/numbers.tex`, and every table body from `paper_assets/tables/`. Re-running the analysis therefore regenerates the numbers in the text, tables and figures together, and they cannot drift apart.

## Experiment index

| Paper element | Produced by | Raw results |
|---|---|---|
| Main comparison, per-dataset table, Wilcoxon, Friedman, critical-difference diagram | `experiments.main`, `analysis.main_part` | `results/main_s*.csv` |
| Ablation | `experiments.abl_default`, `analysis.abl_part` | `results/abl_default*.csv` |
| DE table and figure | `de_search`, `analysis.de_part` | `results/de_*.json` |
| Sensitivity, convergence | `experiments.sens`, `experiments.conv` | `results/sens.csv`, `results/conv.csv` |
| New-client test | `experiments.newclient` | `results/newclient.csv` |
| Corrupted client | `experiments.robust` | `results/robust.csv` |
| Cross-rig federation | `experiments.crossrig` | `results/crossrig.csv` |
| t-SNE, confusion matrices | `experiments.emb`, `experiments.confusion` | `results/emb.pkl`, `results/confusion.pkl` |

## Protocol in brief

* Each recording is split in time. Support shots come from non-overlapping windows lying entirely in the first 25 % of a recording, test windows start after 62 %, and design decisions used windows between 30 % and 58 % with separate seeds.
* A client is one operating condition of one dataset (15 clients). Label-skewed clients label the normal class and a rotating subset of the fault types.
* Seeds 0-9 are reported. Seeds 100-103 (development) and 200-201 (DE fitness) are disjoint from them.

## Reproducibility notes

JAX is run on the CPU with one thread per process (`fpf/__init__.py`), which makes the runs deterministic on a given machine and software stack. Results obtained with other library versions or CPU architectures can differ in the last digits of floating-point arithmetic; `python -m fpf.verify` re-runs three configurations and compares them with the stored results.

## Tests

```bash
pytest -q
```

## Licence

Code: MIT (see `LICENSE`). Datasets: see the original repositories.
