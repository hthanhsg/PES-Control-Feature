# Prediction and Explanation Sensitivity under Controlled Feature Shifts — code and outputs

Code, configuration and record-level outputs for the paper

> *Prediction and Explanation Sensitivity under Controlled Feature Shifts in
> Malware and Phishing Detection.*

The package implements a paired evaluation of feature-attribution stability
with a **ladder of matched controls** (displacement-, probability-, operation-
and MI-rank-matched), a **capability-constrained Android permission shift**,
and a **cross-source URL study** (PhiUSIIL → PhishVN with an in-target
reference), over three datasets, three detector families and ten resplits. It
regenerates every table and figure of the paper. The reporting checklist for
explanation-stability claims is in [`CHECKLIST.md`](CHECKLIST.md).

## Layout

```
rev/                   evaluation library
  data.py              dataset loading, capping, resplits, MI screening
  transforms.py        mask + the control ladder (redistribute, op-, |Δp|-, MI-rank-matched),
                       discrete mask, permission addition
  explainers.py        ablation (deterministic), SHAP variants, LIME/KernelSHAP floors
  metrics.py           tie-aware top-k instability, chance level
  models.py            LR / RF / GB with fixed settings
  source_study.py      main source-benchmark protocol (all stages)
  url_study.py         PhiUSIIL → PhishVN study, fragment edit, in-target models
  tables.py            LaTeX tables, figures, rev_macros.tex, record-level mixed model
  synthetic.py         synthetic data for the smoke test
run_all.py             runs everything; writes results/{source,tables,figures}, environment.json
run_full_matched.py    fully matched control (operation AND |Δp|), incl. disjoint variant
inspect_phishvn.py     prints the columns/values of a PhishVN CSV to fill PhishVNSpec
summarize_url.py       quick textual summary of the URL-study outputs
paper/                 compaction scripts for the conference (LNCS) tables and figures
scripts/               verification against the published numbers
data/                  input CSVs (not redistributed; see data/README.md)
results/               outputs of the official run (in the Zenodo deposit; see results/README.md)
```

## Environment

Python 3.13 with the pinned versions in `requirements.txt` (NumPy 2.1.3,
pandas 2.2.3, scikit-learn 1.6.1, SciPy 1.16.3, SHAP 0.52.0, LIME 0.2.0.1 —
the versions recorded in the official runs).

```bash
conda create -n pes python=3.13 -y && conda activate pes
pip install -r requirements.txt
```

## Configuration to verify before an official run

Four settings encode conventions of the archived pipeline and are marked
`CONFIRM` in the source. They are set to the values used for the published
numbers; verify them if you change datasets or snapshots:

1. `rev/data.py::SOURCES` — label column and threat coding per dataset
   (PhiUSIIL labels are inverted: 1 = legitimate in the raw file).
2. `rev/url_study.py::PhishVNSpec` — column names of the PhishVN CSV
   (override with `--phishvn-spec spec.json`; `inspect_phishvn.py` prints the
   columns of your file).
3. `rev/transforms.py::redistribute` — the coordinate pool of the
   conventional control (default: all d coordinates).
4. `rev/url_study.py::edit_fragment` — the fragment generator
   (`#section-` + 8 random characters).

## Running

```bash
# Smoke test on synthetic data (~3 minutes, no downloads)
python run_all.py --synthetic --quick --out results_smoke

# Full run (~6–8 CPU-hours; UCI sets auto-download, PhishVN CSV required)
python run_all.py --data-dir data --phishvn data/phishvn.csv --out results

# Fully matched control (operation AND |Δp|, plus the disjoint variant)
python run_full_matched.py --data-dir data --out results

# Rebuild tables/figures from saved record-level outputs only
python run_all.py --tables-only --out results
```

Defaults: 10 split seeds, 3 models (LR, RF, GB), every held-out record
explained (`--max-explain N` to cap), SHAP/background explainers on 500
records per split. `results/environment.json` records the library versions.


## Data and licensing

Raw datasets are not redistributed; see [`data/README.md`](data/README.md).
Code is MIT-licensed (see `LICENSE`). Please cite the paper (see
`CITATION.cff`).

## Scope

This package reproduces the paper's numerical results from public data. It
does not include live malware execution, phishing-page interaction, APK-level
transformations or any human-analyst study; numerical masks and URL string
edits are not problem-space validated attacks (see item 8 of the checklist).
