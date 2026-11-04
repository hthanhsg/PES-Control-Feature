#!/usr/bin/env python3
"""Fully matched control: same operation as the mask AND the same probability response.

For each held-out record and severity-two mask, K operation-matched candidates are drawn (same
shrink-towards-median operation on random coordinates, l0 and l2 matched); the candidate whose
|delta p| is closest to that of the mask is kept, and pairs with a remaining gap above --tol are
excluded. Splits and models are rebuilt with the same seeds as run_all.py, so results are directly
comparable with Table "ladder". Primary explainer only (ablation, median reference, probability).

  python run_full_matched.py --data-dir data --out results            (all datasets, 10 seeds)
  python run_full_matched.py --data-dir data --out results --seeds 17 43 --datasets naticusdroid

Writes <out>/source/fullmatch_<dataset>.parquet|csv.gz and <out>/tables/tab_fullmatch.tex.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rev import transforms as T
from rev.common import DATASET_LABEL as DL, MODEL_LABEL as ML, Config, Timer, ci, fmt, seed_t_interval, \
    setup_logging, write_table
from rev.data import cap_table, load_raw, make_split, stratified_subset
from rev.explainers import ablation
from rev.metrics import k_of, topk_instability
from rev.models import make_model, proba


def run(ds: str, cfg: Config, K: int, tol: float, out: Path) -> pd.DataFrame:
    X, y = load_raw(ds, cfg.data_dir)
    X, y = cap_table(X, y, cfg.cap, cfg.cap_seed)
    rows = []
    for seed in cfg.seeds:
        split = make_split(ds, X, y, seed, cfg)
        d = len(split.features)
        k = k_of(d, cfg.k_frac)
        b = split.median_std
        cohort = stratified_subset(split.y_test, cfg.max_explain, seed)
        Z = split.X_test[cohort]
        J = T.mask_coords(split, 2)
        rng = np.random.default_rng([seed, 2, 11])
        Zs, Rs, first = [], [], []
        cand = {"any": ([], []), "disjoint": ([], [])}  # candidate vectors and their owner record
        for i, z in enumerate(Z):                       # candidates do not depend on the model
            ts = T.mask(z, b, J, 2, cfg.move_eps)
            Zs.append(ts.z)
            R = np.flatnonzero(np.abs(ts.z - z) > cfg.move_eps)   # coordinates actually moved by the mask
            Rs.append(R)
            for key, excl in (("any", None), ("disjoint", R)):
                cs = [T.op_matched(z, b, ts.n0, ts.l2, rng, cfg.move_eps, exclude=excl) for _ in range(K)]
                cs = [c.z for c in cs if c.valid]
                if key == "any":
                    first.append(cs[0] if cs else z.copy())  # first valid candidate = plain operation-matched control
                if not cs:
                    cs = [z.copy()]                       # placeholder; flagged invalid below
                cand[key][0].extend(cs)
                cand[key][1].extend([i] * len(cs))
        Zs, Zop = np.vstack(Zs), np.vstack(first)
        pools = {}
        for key, (vecs, own) in cand.items():
            own = np.array(own)
            st = np.r_[0, np.flatnonzero(np.diff(own)) + 1]
            pools[key] = (np.vstack(vecs), st, np.r_[st[1:], len(own)])

        def overlap(zc, z, R):
            S = np.flatnonzero(np.abs(zc - z) > cfg.move_eps)
            return len(np.intersect1d(S, R)) / max(len(R), 1) if len(S) else np.nan
        for kind in cfg.models:
            with Timer(f"full-matched {ds} seed={seed} model={kind}"):
                m = make_model(kind, seed).fit(split.X_train, split.y_train)
                p0 = proba(m, Z)
                ps = proba(m, Zs)
                pop = proba(m, Zop)
                cls = (p0 >= 0.5).astype(int)
                q_s = np.abs(ps - p0)
                A0 = ablation(m, Z, b, cls)
                As, Aop = ablation(m, Zs, b, cls), ablation(m, Zop, b, cls)
                sel = {}
                for key, (C, st, en) in pools.items():
                    pc = proba(m, C)
                    pick, gap = np.empty(len(Z), int), np.empty(len(Z))
                    for i, (a, e) in enumerate(zip(st, en)):
                        g = np.abs(np.abs(pc[a:e] - p0[i]) - q_s[i])
                        j = int(np.argmin(g))
                        pick[i], gap[i] = a + j, g[j]
                    sel[key] = (C[pick], pc[pick], gap, ablation(m, C[pick], b, cls))
                for i in range(len(Z)):
                    rec = dict(dataset=ds, seed=seed, model=kind, record_uid=int(split.idx_test[cohort[i]]),
                               D_s=topk_instability(A0[i], As[i], k, cfg.tie_abs, cfg.tie_rel),
                               D_op=topk_instability(A0[i], Aop[i], k, cfg.tie_abs, cfg.tie_rel),
                               q_s=q_s[i], q_op=abs(pop[i] - p0[i]), n0=len(Rs[i]),
                               ov_op=overlap(Zop[i], Z[i], Rs[i]))
                    for key, tag in (("any", "full"), ("disjoint", "disj")):
                        Zf, pf, gap, Af = sel[key]
                        moved = bool(np.any(np.abs(Zf[i] - Z[i]) > cfg.move_eps))
                        rec.update({f"D_{tag}": topk_instability(A0[i], Af[i], k, cfg.tie_abs, cfg.tie_rel),
                                    f"q_{tag}": abs(pf[i] - p0[i]), f"gap_{tag}": gap[i],
                                    f"valid_{tag}": int(gap[i] <= tol and (moved or len(Rs[i]) == 0)),
                                    f"ov_{tag}": overlap(Zf[i], Z[i], Rs[i])})
                    rows.append(rec)
    df = pd.DataFrame(rows)
    (out / "source").mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out / "source" / f"fullmatch_{ds}.parquet", index=False)
    except Exception:
        df.to_csv(out / "source" / f"fullmatch_{ds}.csv.gz", index=False)
    return df


def table(df: pd.DataFrame, out: Path) -> None:
    rows = []
    for ds in ("naticusdroid", "uci_phishing", "phiusiil"):
        for m in ("lr", "rf", "gb"):
            g = df[(df.dataset == ds) & (df.model == m)]
            if not len(g):
                continue
            op = seed_t_interval((g.D_s - g.D_op).groupby(g.seed).mean())
            cells = [DL[ds], ML[m], ci(*op), fmt(float(g.ov_op.mean()), 2)]
            for tag in ("full", "disj"):
                v = g[g[f"valid_{tag}"] == 1]
                est = seed_t_interval((v.D_s - v[f"D_{tag}"]).groupby(v.seed).mean()) if len(v) else (np.nan,) * 3
                cells += [ci(*est), fmt(float(v[f"ov_{tag}"].mean()), 2) if len(v) else "--",
                          fmt(float(g[f"valid_{tag}"].mean()) * 100, 0)]
            rows.append(cells)
    write_table(out / "tables" / "tab_fullmatch.tex",
                "Severity-two contrast against controls that apply the same operation as the mask: on random "
                "coordinates (Operation), additionally matched on $|\\Delta p|$ within 0.02 (Fully matched), and "
                "matched on $|\\Delta p|$ using only coordinates the mask did not move (Disjoint). Ov.: mean share of "
                "the mask's moved coordinates that the control also moves; \\%: share of records with a valid control.",
                "tab:fullmatch", ["Dataset", "Model", "Operation", "Ov.", "Fully matched", "Ov.", "\\%",
                                  "Disjoint", "Ov.", "\\%"], rows, align="llrrrrrrrr", size="\\scriptsize")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--max-explain", type=int)
    ap.add_argument("--candidates", type=int, default=100)
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()
    cfg = Config(out_dir=a.out, data_dir=a.data_dir)
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
    df = pd.concat([run(ds, cfg, a.candidates, a.tol, out) for ds in cfg.datasets], ignore_index=True)
    table(df, out)
    print((out / "tables" / "tab_fullmatch.tex").read_text())


if __name__ == "__main__":
    main()
