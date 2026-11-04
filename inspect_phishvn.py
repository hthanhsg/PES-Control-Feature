#!/usr/bin/env python3
"""Inspect a PhishVN CSV and write phishvn_spec.json for run_all.py.

Usage:  python inspect_phishvn.py data/phishvn.csv  [--out phishvn_spec.json]
Prints the columns and value counts of likely label/tier columns, guesses the spec, and saves it.
Always check the guess against PhishVN's docs/ column schema before running the URL study.
"""
import argparse
import json
import sys

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--out", default="phishvn_spec.json")
a = ap.parse_args()

df = pd.read_csv(a.csv, low_memory=False)
print(f"{len(df):,} rows, {df.shape[1]} columns\n")
for c in df.columns:
    v = df[c]
    ex = ", ".join(map(str, v.dropna().astype(str).unique()[:4]))
    print(f"  {c:<32} {str(v.dtype):<10} nunique={v.nunique():<8} e.g. {ex[:80]}")


def pick(cands, exclude=()):
    low = {c.lower(): c for c in df.columns if c not in exclude}
    for k in cands:
        for lc, c in low.items():
            if k == lc:
                return c
    for k in cands:
        for lc, c in low.items():
            if k in lc:
                return c
    return None


url = pick(["url", "uri", "link"])
label = pick(["label", "is_phishing", "class", "target", "y"], exclude=[url])
tier = pick(["confidence_tier", "label_confidence", "tier", "confidence"], exclude=[url, label])
time = pick(["first_seen", "timestamp", "collected_at", "date", "time", "seen"], exclude=[url, label, tier])

print()
for name, col in (("label", label), ("tier", tier)):
    if col:
        print(f"{name} column '{col}':\n{df[col].value_counts(dropna=False).head(10).to_string()}\n")

threat = []
if label:
    vals = [str(v) for v in df[label].dropna().unique()]
    threat = [v for v in vals if v.lower() in ("phishing", "phish", "malicious", "1", "true", "1.0")]
spec = {"path": a.csv, "url_col": url, "label_col": label, "threat_values": threat,
        "tier_col": tier, "keep_tiers": ["gold", "silver"], "time_col": time}
print("Guessed spec:\n" + json.dumps(spec, indent=2, ensure_ascii=False))
missing = [k for k in ("url_col", "label_col") if not spec[k]] + ([] if threat else ["threat_values"])
if missing:
    print(f"\n!! Could not determine {missing}; edit {a.out} by hand.", file=sys.stderr)
if tier and not set(df[tier].astype(str).str.lower()) & {"gold", "silver"}:
    print(f"\n!! Tier column '{tier}' has no 'gold'/'silver' values; set keep_tiers in {a.out}.", file=sys.stderr)
with open(a.out, "w", encoding="utf-8") as fh:
    json.dump(spec, fh, indent=2, ensure_ascii=False)
print(f"\nSaved {a.out}. Check it, then run:\n  python run_all.py --data-dir data --out results --skip-source "
      f"--skip-stochastic --phishvn {a.csv} --phishvn-spec {a.out}")
