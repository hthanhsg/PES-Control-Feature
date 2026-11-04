# Results (record-level outputs)

This directory is empty in the git repository. The Zenodo deposit adds the
outputs of the official run here, as promised in the paper ("code,
configuration and record-level outputs"):

- `source/*.parquet` (or `.csv.gz`) — one row per explained record, split
  seed, model and condition: the paired contrast components, probability
  responses, validity flags and match gaps.
- `tables/*.tex` — all generated LaTeX tables, including `rev_macros.tex`.
- `figures/*.pdf` — all generated figures.
- `environment.json` — Python and library versions of the run.

Regenerate everything with `python run_all.py --data-dir data --out results`
(about 6–8 CPU-hours), then `python run_full_matched.py --data-dir data --out
results` for the fully matched control, and finally the scripts in `paper/`
for the camera-ready tables and figures.
