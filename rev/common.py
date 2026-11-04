"""Shared utilities: configuration, uncertainty summaries, LaTeX table writing."""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

LOG = logging.getLogger("rev")

# Seeds 1-5 are the original manuscript seeds; 6-10 are the additional resplits (R1-M6).
ORIGINAL_SEEDS = (17, 43, 101, 211, 307)
EXTRA_SEEDS = (509, 613, 727, 829, 937)

# Name substrings of the original eligibility rule (Sec. 4.1 of the manuscript).
KEYWORDS = ("permission", "api", "url", "domain", "ssl", "html", "script", "redirect", "web",
            "socket", "address", "ip", "form", "request", "sms", "contact", "login", "bank",
            "https", "email")

MASK_FRACTIONS = (0.16, 0.30, 0.46)
MASK_STRENGTHS = (0.35, 0.55, 0.75)


@dataclass
class Config:
    out_dir: str = "results"
    data_dir: str = "data"
    seeds: tuple = ORIGINAL_SEEDS + EXTRA_SEEDS
    models: tuple = ("lr", "rf", "gb")
    datasets: tuple = ("naticusdroid", "uci_phishing", "phiusiil")
    cap: int = 10_000
    cap_seed: int = 0
    max_features: int = 24
    test_size: float = 0.30
    max_explain: int | None = None          # None = full held-out partition (R1-M3)
    original_cohort: int = 40               # size of the original 40-case cohort, kept for comparison
    k_frac: float = 0.15
    k_fracs_sensitivity: tuple = (0.10, 0.15, 0.25, 0.33)
    tie_abs: float = 1e-9                   # harmonized tie rule (R1-M5): abs + rel * max|a|
    tie_rel: float = 1e-6
    move_eps: float = 1e-7                  # realized-coordinate threshold (Sec. 4.1)
    l2_tol: float = 2e-5
    dp_candidates: int = 200                # candidate controls for the probability-matched control (R1-M2)
    dp_tol: float = 0.02
    background_n: int = 100
    background_explain_n: int = 500         # records used for background-interventional ablation (costly)
    stochastic_cases: int = 100             # per dataset x model x seed (R1-M3)
    stochastic_budgets: tuple = (128, 512, 2048, "default")
    stochastic_seeds: tuple = ORIGINAL_SEEDS
    bootstrap: int = 2000
    quick: bool = False

    def save(self, path: Path) -> None:
        d = asdict(self)
        path.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "run.log")])
    for noisy in ("shap", "lime", "matplotlib", "statsmodels"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore")


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        LOG.info("%s: %.1fs", self.label, time.time() - self.t0)


# ----------------------------------------------------------------------------- uncertainty

def seed_t_interval(seed_means: Sequence[float], level: float = 0.95) -> tuple[float, float, float]:
    """Mean and descriptive Student-t interval over seed-level means (Eq. 12)."""
    x = np.asarray([v for v in seed_means if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    m = float(x.mean())
    if len(x) < 2:
        return (m, np.nan, np.nan)
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return (m, m - h, m + h)


def cluster_bootstrap(df: pd.DataFrame, value: str, seed_col: str = "seed",
                      record_col: str = "record_uid", B: int = 2000, rng_seed: int = 0,
                      level: float = 0.95) -> tuple[float, float, float]:
    """Two-level bootstrap: resample seeds, then records within each sampled seed (R1-M6).

    The statistic is the mean of seed-level means, i.e. the same estimand as Table 5.
    """
    rng = np.random.default_rng(rng_seed)
    groups = {s: g[value].to_numpy(dtype=float) for s, g in df.groupby(seed_col)}
    groups = {s: v[np.isfinite(v)] for s, v in groups.items() if np.isfinite(v).any()}
    if not groups:
        return (np.nan, np.nan, np.nan)
    seeds = list(groups)
    point = float(np.mean([groups[s].mean() for s in seeds]))
    stats_b = np.empty(B)
    for b in range(B):
        chosen = rng.choice(seeds, size=len(seeds), replace=True)
        means = []
        for s in chosen:
            v = groups[s]
            means.append(v[rng.integers(0, len(v), len(v))].mean())
        stats_b[b] = np.mean(means)
    lo, hi = np.quantile(stats_b, [(1 - level) / 2, 1 - (1 - level) / 2])
    return (point, float(lo), float(hi))


# ----------------------------------------------------------------------------- LaTeX output

def fmt(x, nd: int = 3, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", "{,}")
    v = float(x)
    if abs(v) < 0.5 * 10 ** (-nd):
        v = 0.0
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return s.replace("-", "$-$") if not signed else s.replace("-", "$-$").replace("+", "$+$")


CI_SEP = "\u00a7"   # marks "estimate | interval"; write_table puts the interval on its own row


def ci(m, lo, hi, nd: int = 3) -> str:
    """Point estimate and interval; the interval is typeset on a second row to keep tables narrow."""
    return f"{fmt(m, nd, True)}{CI_SEP}{{\\tiny[{fmt(lo, nd, True)}, {fmt(hi, nd, True)}]}}"


def write_table(path: Path, caption: str, label: str, header: Sequence[str], rows: Iterable[Sequence[str]],
                align: str | None = None, note: str | None = None, size: str = "\\small") -> None:
    """Write a complete table environment that the manuscript \\input's."""
    rows = list(rows)
    align = align or ("l" * 2 + "r" * (len(header) - 2))
    lines = ["% Auto-generated by revision experiments; do not edit by hand.",
             "\\begin{table}[!htbp]", f"\\caption{{{caption}}}\\label{{{label}}}", "\\centering" + size,
             "\\setlength{\\tabcolsep}{2.5pt}",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + "\\\\", "\\midrule"]
    for r in rows:
        if any(CI_SEP in c for c in r):
            lines.append(" & ".join(c.split(CI_SEP)[0] for c in r) + "\\\\")
            lines.append(" & ".join(c.split(CI_SEP)[1] if CI_SEP in c else "" for c in r) + "\\\\[2pt]")
        else:
            lines.append(" & ".join(r) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\par\\smallskip{{\\footnotesize {note}}}")
    lines.append("\\end{table}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_macros(path: Path, macros: dict[str, str]) -> None:
    """Numbers quoted in the prose are emitted as macros, e.g. \\RevOpMatchedNatRF."""
    lines = ["% Auto-generated numeric macros used in the revised prose."]
    for k, v in sorted(macros.items()):
        lines.append(f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


DATASET_LABEL = {"naticusdroid": "NATICUSdroid", "uci_phishing": "UCI Phishing", "phiusiil": "PhiUSIIL"}
MODEL_LABEL = {"lr": "LR", "rf": "RF", "gb": "GB"}
