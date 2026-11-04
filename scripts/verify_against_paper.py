#!/usr/bin/env python3
"""Compare regenerated compact tables with the numbers printed in the paper.

Usage:  python scripts/verify_against_paper.py <dir-with-tab_*_c.tex> [--tol 0.005]

<dir> must contain tab_ladder_c.tex, tab_capability_c.tex and tab_url_c.tex as
written by paper/make_tables.py. Every numeric cell is compared with the value
printed in the conference paper (Tables 1-3); deviations above --tol are
reported and the exit code is non-zero if any cell fails.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- expected values
# Conference Table 2 (tab:ladder): Redistr., |dp|-matched (match share %), Operation, MI rank, gamma=0
LADDER = {
    ("NATICUSdroid", "LR"): (-0.285, -0.167, 98, +0.006, +0.013, -0.283),
    ("NATICUSdroid", "RF"): (+0.504, +0.084, 52, +0.201, +0.365, +0.294),
    ("NATICUSdroid", "GB"): (+0.507, +0.072, 51, +0.195, +0.355, +0.376),
    ("UCI Phishing", "LR"): (+0.027, -0.017, 83, +0.053, +0.061, -0.019),
    ("UCI Phishing", "RF"): (+0.368, +0.035, 73, +0.284, +0.322, +0.258),
    ("UCI Phishing", "GB"): (+0.362, +0.057, 73, +0.270, +0.317, +0.282),
    ("PhiUSIIL", "LR"):     (+0.060, +0.062, 100, +0.046, +0.064, +0.060),
    ("PhiUSIIL", "RF"):     (-0.105, +0.001, 100, -0.035, +0.000, +0.000),
    ("PhiUSIIL", "GB"):     (-0.014, -0.004, 100, -0.008, -0.004, +0.000),
}
# Conference Table 3 (tab:capability): delta(1,2,4), evasion(1,2,4), random(1,2,4)
CAPABILITY = {
    "LR": (+0.058, +0.124, +0.167, 0.561, 0.871, 0.953, 0.127, 0.210, 0.340),
    "RF": (+0.073, +0.062, +0.060, 0.625, 0.692, 0.958, 0.056, 0.098, 0.189),
    "GB": (+0.152, +0.173, +0.123, 0.569, 0.655, 0.958, 0.070, 0.135, 0.261),
}
# Conference Table 1 (tab:url): source BA; PhishVN BA, AUC, FPR, flagged; in-target BA, AUC;
# fragment evasions, false alarms
URL = {
    "LR": (0.842, 0.526, 0.722, 0.009, 0.016, 0.780, 0.882, 0, 1000),
    "RF": (0.846, 0.565, 0.588, 0.638, 0.657, 0.762, 0.882, 0, 986),
}

NUM = re.compile(r"[-+]?\d*\.?\d+")


def numbers(cell: str) -> list[float]:
    cell = cell.replace("$", "").replace("{,}", "").replace("\\%", "").replace("\\", "")
    cell = cell.replace("−", "-").replace("+", "+")
    return [float(x) for x in NUM.findall(cell)]


def rows(path: Path) -> list[list[str]]:
    body = path.read_text().split("\\midrule")[1].split("\\bottomrule")[0]
    out = []
    for line in body.strip().split("\n"):
        line = re.sub(r"\\\\(\[\d+pt\])?$", "", line.strip())
        if line and "&" in line:
            out.append([c.strip() for c in line.split("&")])
    return out


def check(name: str, got: list[float], exp: tuple, tol: float, bad: list) -> None:
    if len(got) != len(exp):
        bad.append(f"{name}: expected {len(exp)} numbers, parsed {len(got)}: {got}")
        return
    for g, e in zip(got, exp):
        t = 0.5 if abs(e) >= 5 else tol          # percentages / counts get an absolute 0.5
        if abs(e) >= 5 and float(e).is_integer():
            t = max(t, 0.5)
        if abs(g - e) > t:
            bad.append(f"{name}: got {g} expected {e} (tol {t})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tabdir", type=Path, nargs="?",
                    default=Path(__file__).resolve().parent.parent / "paper")
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args()
    missing = [f for f in ("tab_ladder_c.tex", "tab_capability_c.tex", "tab_url_c.tex")
               if not (a.tabdir / f).is_file()]
    if missing:
        sys.exit(f"Missing in {a.tabdir}: {', '.join(missing)}.\n"
                 "Generate them first:  python paper/make_tables.py <dir-with-tab_ladder.tex>")
    bad: list[str] = []

    prev = None
    for r in rows(a.tabdir / "tab_ladder_c.tex"):
        ds = r[0] or prev
        prev = ds
        got = [x for c in r[2:] for x in numbers(c)]
        check(f"ladder {ds}/{r[1]}", got, LADDER[(ds, r[1])], a.tol, bad)

    for r in rows(a.tabdir / "tab_capability_c.tex"):
        got = [x for c in r[1:] for x in numbers(c)]
        check(f"capability {r[0]}", got, CAPABILITY[r[0]], a.tol, bad)

    for r in rows(a.tabdir / "tab_url_c.tex"):
        got = [x for c in r[1:] for x in numbers(c)]
        check(f"url {r[0]}", got, URL[r[0]], a.tol, bad)

    if bad:
        print(f"FAILED: {len(bad)} cell(s) deviate from the published values")
        for b in bad:
            print("  " + b)
        return 1
    n = len(LADDER) * 6 + len(CAPABILITY) * 9 + len(URL) * 9
    print(f"OK: all {n} cells match the published values within tolerance {a.tol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
