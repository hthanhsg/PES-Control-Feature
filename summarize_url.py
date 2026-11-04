#!/usr/bin/env python3
"""Print the key URL-study numbers of one output folder (e.g. the host-only run).

  python summarize_url.py results_host
"""
import sys
from pathlib import Path

import pandas as pd

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results") / "url"
perf = pd.concat([pd.read_csv(d / "performance.csv"), pd.read_csv(d / "in_target.csv")], ignore_index=True)
cols = ["n", "ba", "auc", "pr_auc", "fpr", "pred_pos_rate", "brier"]
print(perf.groupby(["model", "population"])[cols].mean().round(3).to_string())
fr = pd.read_csv(d / "fragment.csv")
fr = fr[fr.target == "predicted"]
print("\nFragment edits (predicted-class explanations):")
for m, g in fr.groupby("model"):
    ev = ((g.direction == "threat->benign") & (g.y == 1)).sum()
    fa = ((g.direction == "benign->threat") & (g.y == 0)).sum()
    print(f"  {m}: rows={len(g)} evasions={ev} false_alarms={fa} delta={(g.D_frag - g.D_ctrl).mean():+.3f}")
