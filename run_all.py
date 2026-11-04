#!/usr/bin/env python3
"""Run every revision analysis and write the LaTeX tables/figures used by the revised manuscript.

Examples
  # full run on the server (all datasets, 10 seeds, full held-out partitions)
  python run_all.py --data-dir data --phishvn data/phishvn.csv --out results

  # only the source study with the original five seeds
  python run_all.py --skip-url --seeds 17 43 101 211 307

  # rebuild tables from existing record files
  python run_all.py --tables-only

  # smoke test on synthetic data (minutes)
  python run_all.py --synthetic --quick --out results_smoke

Copy (or symlink) <out>/tables and <out>/figures into the manuscript folder as
results/tables and results/figures; the manuscript \\input's them automatically.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path

from rev import __version__
from rev.common import Config, Timer, setup_logging
from rev import source_study, tables, url_study

LOG = logging.getLogger("rev")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--phishvn", default="data/phishvn.csv")
    ap.add_argument("--phishvn-spec", help="JSON file overriding PhishVNSpec fields (column names, label values)")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--max-explain", type=int, help="cap on explained held-out records per split (default: all)")
    ap.add_argument("--skip-source", action="store_true")
    ap.add_argument("--skip-stochastic", action="store_true")
    ap.add_argument("--skip-url", action="store_true")
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny cohorts for a smoke test")
    ap.add_argument("--url-host-only", action="store_true",
                    help="reduce PhiUSIIL URLs to host names before training (format-matched to domain-only PhishVN)")
    ap.add_argument("--synthetic", action="store_true", help="generate synthetic stand-in data first")
    a = ap.parse_args()

    cfg = Config(out_dir=a.out, data_dir=a.data_dir, quick=a.quick)
    if a.seeds:
        cfg.seeds = tuple(a.seeds)
    if a.datasets:
        cfg.datasets = tuple(a.datasets)
    if a.models:
        cfg.models = tuple(a.models)
    if a.max_explain:
        cfg.max_explain = a.max_explain
    out = Path(a.out)
    setup_logging(out)
    cfg.save(out / "config.json")
    import numpy, pandas, scipy, sklearn
    env = dict(version=__version__, python=sys.version.split()[0], platform=platform.platform(),
               numpy=numpy.__version__, pandas=pandas.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    from importlib import metadata
    for mod in ("shap", "lime", "statsmodels", "tldextract"):
        try:
            env[mod] = metadata.version(mod)
        except Exception:
            try:
                __import__(mod)
                env[mod] = "unknown version"
            except Exception:
                env[mod] = "missing"
    (out / "environment.json").write_text(json.dumps(env, indent=2))
    LOG.info("environment: %s", env)

    spec = url_study.PhishVNSpec(path=a.phishvn)
    if a.synthetic:
        from rev.synthetic import write_all
        spec = write_all(Path(cfg.data_dir))
    elif a.phishvn_spec:
        spec = url_study.PhishVNSpec(**{**spec.__dict__, **json.loads(Path(a.phishvn_spec).read_text())})

    skip_url = a.skip_url
    if not (a.tables_only or skip_url or a.synthetic) and not Path(spec.path).is_file():
        LOG.warning("PhishVN file not found: %s -> the URL study is SKIPPED in this run. "
                    "Download PhishVN (dataset_url.csv), run `python inspect_phishvn.py <file>` and rerun with "
                    "--skip-source --skip-stochastic --phishvn <file> --phishvn-spec phishvn_spec.json", spec.path)
        skip_url = True

    if not a.tables_only:
        if not a.skip_source:
            for ds in cfg.datasets:
                with Timer(f"source study {ds}"):
                    source_study.run_dataset(ds, cfg, out)
        if not a.skip_stochastic:
            for ds in cfg.datasets:
                with Timer(f"stochastic floors {ds}"):
                    source_study.run_stochastic(ds, cfg, out)
        if not skip_url:
            try:
                with Timer("URL study"):
                    url_study.run(cfg, spec, out, frag_per_seed=20 if cfg.quick else 200,
                                  attr_per_class=(8,) if cfg.quick else (24, 200), host_only=a.url_host_only)
            except Exception:
                LOG.exception("URL study failed; source-study tables are still built. "
                              "Fix the PhishVN spec and rerun with --skip-source --skip-stochastic.")
    if not any((out / "source").glob("records_*")):
        LOG.info("no source-study records in %s: tables skipped (use summarize_url.py for URL-only runs)", out)
        return
    with Timer("tables"):
        tables.build(out, cfg)


if __name__ == "__main__":
    main()
