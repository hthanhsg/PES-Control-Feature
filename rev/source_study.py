"""Source-study runs: paired transforms, the control ladder, all explainers, record-level output.

Writes <out>/source_records.parquet (or .csv.gz) with one row per
(dataset, seed, model, explainer config, family, severity, control, record).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from . import transforms as T
from .common import Config, Timer
from .data import cap_table, load_raw, make_split, stratified_subset
from .explainers import ShapExplainer, ablation, ablation_background
from .metrics import (cutoff_tied, k_of, sign_flip_rate, signed_cosine, topk_instability, topk_weights)
from .models import make_model, proba

LOG = logging.getLogger("rev")

# (name, baseline, space). The first entry is the primary configuration of Table 5.
EXPLAINERS = [
    ("ablation", "median", "prob"),
    ("ablation", "median", "logit"),
    ("ablation", "mean", "prob"),
    ("ablation", "background", "prob"),
    ("shap", "background", "native"),
]
PRIMARY = EXPLAINERS[0]


def in_scope(cfg_e, fam: str, sev: int) -> bool:
    """Which (family, severity) cells each explainer is run on; keeps the full run tractable."""
    name, base, space = cfg_e
    if cfg_e == PRIMARY:
        return True
    if name == "ablation" and base == "mean":
        return fam == "mask"                                   # all severities (Table severity-full)
    if name == "ablation" and base == "background":
        return fam == "mask" and sev == 2
    if name == "ablation" and space == "logit":
        return fam == "mask" and sev == 2
    return (fam, sev) in (("mask", 2), ("add_perm", 2))       # Linear/TreeSHAP


def redistribute_batch(z: np.ndarray, delta: np.ndarray, K: int, rng, eps: float) -> np.ndarray:
    """K independent redistribution controls at once (same construction as transforms.redistribute)."""
    r = np.abs(delta) > eps
    mags = np.abs(delta[r])
    n0, d = len(mags), len(z)
    out = np.repeat(z[None, :], K, axis=0)
    if n0 == 0:
        return out
    pos = np.argsort(rng.random((K, d)), axis=1)[:, :n0]
    vals = rng.permuted(np.tile(mags, (K, 1)), axis=1) * rng.choice([-1.0, 1.0], size=(K, n0))
    np.put_along_axis(out, pos, np.take_along_axis(out, pos, axis=1) + vals, axis=1)
    return out
CONTROLS = ("redistribute", "dp_matched", "op_matched", "mi_matched")
FAMILY_ID = {"mask": 1, "mask_discrete": 2, "add_perm": 3}


def _explain(cfg_e, model, kind, Z, cls, split, B, shap_cache):
    name, base, space = cfg_e
    if name == "ablation":
        if base == "median":
            return ablation(model, Z, split.median_std, cls, space)
        if base == "mean":
            return ablation(model, Z, split.mean_std, cls, space)
        return ablation_background(model, Z, B, cls, space)
    if kind not in shap_cache:
        shap_cache[kind] = ShapExplainer(model, kind, B)
    return shap_cache[kind](Z, cls)


def _group_instability(a, a2, clusters, k_frac, tie_abs, tie_rel):
    ga = np.array([np.abs(a[c]).sum() for c in clusters])
    gb = np.array([np.abs(a2[c]).sum() for c in clusters])
    return topk_instability(ga, gb, k_of(len(clusters), k_frac), tie_abs, tie_rel)


def feature_clusters(split, thr: float = 0.8) -> list[np.ndarray]:
    """Correlation clusters (|Spearman| >= thr, average linkage) for grouped attributions (R2-M5, R3-M3)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr
    X = split.X_train
    if X.shape[1] < 2:
        return [np.arange(X.shape[1])]
    rho = np.nan_to_num(spearmanr(X).statistic, nan=0.0)
    rho = np.atleast_2d(rho)
    dist = 1 - np.abs(rho)
    np.fill_diagonal(dist, 0)
    lab = fcluster(linkage(squareform(dist, checks=False), "average"), t=1 - thr, criterion="distance")
    return [np.flatnonzero(lab == c) for c in np.unique(lab)]


def run_dataset(name: str, cfg: Config, out: Path) -> None:
    X, y = load_raw(name, cfg.data_dir)
    X, y = cap_table(X, y, cfg.cap, cfg.cap_seed)
    rows, clean_rows, elig_rows, red_rows, case_rows = [], [], [], [], []
    seeds = cfg.seeds[:2] if cfg.quick else cfg.seeds
    for seed in seeds:
        split = make_split(name, X, y, seed, cfg)
        d = len(split.features)
        k = k_of(d, cfg.k_frac)
        clusters = feature_clusters(split)
        rng_bg = np.random.default_rng(seed)
        B = split.X_train[rng_bg.choice(len(split.X_train), size=min(cfg.background_n, len(split.X_train)), replace=False)]
        # ---- eligibility and redundancy reports (R1-M1 iv, R2-M5)
        for sev in (1, 2, 3):
            J = T.mask_coords(split, sev)
            elig_rows.append(dict(dataset=name, seed=seed, severity=sev, d=d, n_eligible=int(split.eligible.sum()),
                                  masked=";".join(split.features[j] for j in J), n_masked=len(J)))
        from sklearn.metrics import roc_auc_score as _auc
        sauc = [max(_auc(split.y_train, split.X_train[:, j]), 1 - _auc(split.y_train, split.X_train[:, j]))
                for j in range(d)]
        red_rows.append(dict(dataset=name, seed=seed, d=d, n_clusters=len(clusters),
                             max_single_auc=float(np.max(sauc)), n_single_auc_095=int(np.sum(np.array(sauc) >= 0.95)),
                             largest_cluster=int(max(len(c) for c in clusters)),
                             features=";".join(split.features)))
        cohort = stratified_subset(split.y_test, cfg.max_explain if not cfg.quick else 60, seed)
        orig40 = set(stratified_subset(split.y_test, cfg.original_cohort, seed).tolist())
        Zc = split.X_test[cohort]
        yc = split.y_test[cohort]
        for kind in cfg.models:
            with Timer(f"{name} seed={seed} model={kind}"):
                model = make_model(kind, seed).fit(split.X_train, split.y_train)
                p_test = proba(model, split.X_test)
                yhat = (p_test >= 0.5).astype(int)
                tn, fp, fn, tp = confusion_matrix(split.y_test, yhat, labels=[0, 1]).ravel()
                p0 = p_test[cohort]
                cls = (p0 >= 0.5).astype(int)
                clean_rows.append(dict(dataset=name, seed=seed, model=kind, n_test=len(split.y_test),
                                       prevalence=float(split.y_test.mean()),
                                       ba=balanced_accuracy_score(split.y_test, yhat), f1=f1_score(split.y_test, yhat),
                                       auc=roc_auc_score(split.y_test, p_test),
                                       precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1),
                                       cohort_n=len(cohort), cohort_acc=float((cls == yc).mean()),
                                       saturated=float(np.mean((p0 < 1e-3) | (p0 > 1 - 1e-3)))))
                shap_cache: dict = {}
                clean_cache: dict = {}
                fams = [("mask", s) for s in (1, 2, 3)] + [("mask_discrete", 2)]
                if split.binary.mean() > 0.5:
                    fams += [("add_perm", s) for s in (1, 2, 3)]
                for fam, sev in fams:
                    rng = np.random.default_rng([seed, sev, FAMILY_ID[fam], 7])
                    Zs, info_s, ctrl = [], [], {c: [] for c in CONTROLS}
                    J = T.mask_coords(split, sev) if fam != "add_perm" else None
                    for r_i, row in enumerate(cohort):
                        z = Zc[r_i]
                        if fam == "mask":
                            ts = T.mask(z, split.median_std, J, sev, cfg.move_eps)
                        elif fam == "mask_discrete":
                            ts = T.mask_discrete(split, row, J, sev, cfg.move_eps)
                        else:
                            ts = T.add_binary(split, row, (1, 2, 4)[sev - 1], rng, True, cfg.move_eps)
                        Zs.append(ts.z)
                        info_s.append(ts)
                        delta = ts.z - z
                        if fam == "add_perm":
                            c_add = T.add_binary(split, row, (1, 2, 4)[sev - 1], rng, False, cfg.move_eps)
                            for c in CONTROLS:
                                ctrl[c].append(c_add if c == "redistribute" else T.TResult(z.copy(), False, 0, 0.0, "n/a"))
                            continue
                        ctrl["redistribute"].append(T.redistribute(z, delta, rng, cfg.move_eps, l2_tol=cfg.l2_tol))
                        Jr = J[np.abs(delta[J]) > cfg.move_eps] if J is not None else np.array([], int)
                        ctrl["op_matched"].append(T.op_matched(z, split.median_std, ts.n0, ts.l2, rng, cfg.move_eps))
                        ctrl["mi_matched"].append(T.mi_matched(z, split.median_std, Jr, ts.l2, split.mi_rank,
                                                               cfg.move_eps, J))
                        ctrl["dp_matched"].append(None)
                    Zs = np.vstack(Zs)
                    ps = proba(model, Zs)
                    # probability-matched control, batched over records (R1-M2)
                    if fam != "add_perm":
                        cand_Z, owner = [], []
                        Kc = cfg.dp_candidates if not cfg.quick else 30
                        for r_i in range(len(cohort)):
                            cand_Z.append(redistribute_batch(Zc[r_i], Zs[r_i] - Zc[r_i], Kc, rng, cfg.move_eps))
                            owner.append(np.full(Kc, r_i))
                        cand_Z = np.vstack(cand_Z)
                        owner = np.concatenate(owner)
                        cand_p = proba(model, cand_Z)
                        target = np.abs(ps - p0)
                        for r_i in range(len(cohort)):
                            sl = slice(r_i * Kc, (r_i + 1) * Kc)
                            gaps = np.abs(np.abs(cand_p[sl] - p0[r_i]) - target[r_i])
                            i = int(np.argmin(gaps))
                            zc = cand_Z[sl][i]
                            dd = zc - Zc[r_i]
                            ctrl["dp_matched"][r_i] = T.TResult(zc, bool(gaps[i] <= cfg.dp_tol),
                                                                int((np.abs(dd) > cfg.move_eps).sum()),
                                                                float(np.linalg.norm(dd)), f"gap={gaps[i]:.4f}")
                    Zctrl = {c: np.vstack([t.z for t in ctrl[c]]) for c in CONTROLS}
                    pctrl = {c: proba(model, Zctrl[c]) for c in CONTROLS}
                    for cfg_e in EXPLAINERS:
                        if not in_scope(cfg_e, fam, sev):
                            continue
                        sel = np.arange(len(cohort))
                        if cfg_e[1] == "background":            # background ablation and SHAP are costly
                            sel = sel[: cfg.background_explain_n]
                        if cfg_e not in clean_cache:
                            clean_cache[cfg_e] = _explain(cfg_e, model, kind, Zc[sel], cls[sel], split, B, shap_cache)
                        A0 = clean_cache[cfg_e]
                        As = _explain(cfg_e, model, kind, Zs[sel], cls[sel], split, B, shap_cache)
                        Ac = {c: _explain(cfg_e, model, kind, Zctrl[c][sel], cls[sel], split, B, shap_cache)
                              for c in CONTROLS if fam != "add_perm" or c == "redistribute"}
                        primary = cfg_e == PRIMARY
                        for t_i, r_i in enumerate(sel):
                            a0, a_s = A0[t_i], As[t_i]
                            Ds = topk_instability(a0, a_s, k, cfg.tie_abs, cfg.tie_rel)
                            base = dict(dataset=name, seed=seed, model=kind, explainer=cfg_e[0], baseline=cfg_e[1],
                                        space=cfg_e[2], family=fam, severity=sev,
                                        record_uid=int(split.idx_test[cohort[r_i]]), in_orig40=int(cohort[r_i] in orig40),
                                        y=int(yc[r_i]), cls=int(cls[r_i]), p0=float(p0[r_i]), p_s=float(ps[r_i]),
                                        n0_s=info_s[r_i].n0, l2_s=info_s[r_i].l2, D_s=Ds,
                                        cos_s=signed_cosine(a0, a_s),
                                        sf_s=sign_flip_rate(a0, a_s, k, cfg.tie_abs, cfg.tie_rel),
                                        tie_clean=int(cutoff_tied(a0, k, cfg.tie_abs, cfg.tie_rel)), k=k, d=d)
                            if primary:
                                for kf in cfg.k_fracs_sensitivity:
                                    base[f"D_s_k{int(round(kf * 100))}"] = topk_instability(a0, a_s, k_of(d, kf), cfg.tie_abs, cfg.tie_rel)
                                base["D_s_group"] = _group_instability(a0, a_s, clusters, cfg.k_frac, cfg.tie_abs, cfg.tie_rel)
                            for c, Acc in Ac.items():
                                t = ctrl[c][r_i]
                                a_c = Acc[t_i]
                                pc = float(pctrl[c][r_i])
                                rec = dict(base, control=c, valid_c=int(t.valid), n0_c=t.n0, l2_c=t.l2, p_c=pc,
                                           D_c=topk_instability(a0, a_c, k, cfg.tie_abs, cfg.tie_rel),
                                           cos_c=signed_cosine(a0, a_c),
                                           sf_c=sign_flip_rate(a0, a_c, k, cfg.tie_abs, cfg.tie_rel), info_c=t.info)
                                if primary:
                                    for kf in cfg.k_fracs_sensitivity:
                                        rec[f"D_c_k{int(round(kf * 100))}"] = topk_instability(a0, a_c, k_of(d, kf), cfg.tie_abs, cfg.tie_rel)
                                    rec["D_c_group"] = _group_instability(a0, a_c, clusters, cfg.k_frac, cfg.tie_abs, cfg.tie_rel)
                                rows.append(rec)
                        # ---- case study (R2-M6): first seed, RF, primary explainer, severity-two mask
                        if primary and fam == "mask" and sev == 2 and kind == "rf" and seed == seeds[0]:
                            Ds_all = np.array([topk_instability(A0[i], As[i], k, cfg.tie_abs, cfg.tie_rel) for i in range(len(sel))])
                            picks = [int(np.argmax(Ds_all)), int(np.argmin(Ds_all + (p0[sel] < 0.5) * 9))]
                            for pi in picks:
                                for which, A, pp in (("clean", A0, p0), ("mask", As, ps)):
                                    top = np.argsort(-np.abs(A[pi]))[:k]
                                    case_rows.append(dict(dataset=name, seed=seed, record_uid=int(split.idx_test[cohort[pi]]),
                                                          y=int(yc[pi]), input=which, p=float(pp[pi]), D=float(Ds_all[pi]),
                                                          top=";".join(f"{split.features[j]}={A[pi][j]:+.4f}" for j in top)))
    sfx = out / "source"
    sfx.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    for c in ("dataset", "model", "explainer", "baseline", "space", "family", "control", "info_c"):
        if c in df:
            df[c] = df[c].astype("category")
    try:
        df.to_parquet(sfx / f"records_{name}.parquet", index=False)
    except Exception:
        df.to_csv(sfx / f"records_{name}.csv.gz", index=False)
    pd.DataFrame(clean_rows).to_csv(sfx / f"clean_{name}.csv", index=False)
    pd.DataFrame(elig_rows).to_csv(sfx / f"eligibility_{name}.csv", index=False)
    pd.DataFrame(red_rows).to_csv(sfx / f"redundancy_{name}.csv", index=False)
    pd.DataFrame(case_rows).to_csv(sfx / f"cases_{name}.csv", index=False)
    LOG.info("%s: %d record rows", name, len(df))


def run_stochastic(name: str, cfg: Config, out: Path) -> None:
    """Identity floors of LIME and KernelSHAP with defaults and more cases (R1-M3)."""
    import shap
    from .explainers import kernel_shap_attr, lime_attr
    X, y = load_raw(name, cfg.data_dir)
    X, y = cap_table(X, y, cfg.cap, cfg.cap_seed)
    rows = []
    seeds = cfg.stochastic_seeds[:1] if cfg.quick else cfg.stochastic_seeds
    budgets = (128, "default") if cfg.quick else cfg.stochastic_budgets
    n_cases = 4 if cfg.quick else cfg.stochastic_cases
    for seed in seeds:
        split = make_split(name, X, y, seed, cfg)
        k = k_of(len(split.features), cfg.k_frac)
        cohort = stratified_subset(split.y_test, n_cases, seed + 1)
        bg = shap.kmeans(split.X_train, 10)
        J = T.mask_coords(split, 2)
        for kind in ("lr", "rf"):
            model = make_model(kind, seed).fit(split.X_train, split.y_train)
            for r_i in cohort:
                z = split.X_test[r_i]
                c = int(proba(model, z)[0] >= 0.5)
                zs = T.mask(z, split.median_std, J, 2, cfg.move_eps).z
                for bud in budgets:
                    for ex_name, fn in (("LIME", lambda zz, s: lime_attr(model, zz, split.X_train, c, bud, s, split.features)),
                                        ("KernelSHAP", lambda zz, s: kernel_shap_attr(model, zz, bg, c, bud, s))):
                        a1, a2, a3 = fn(z, 1), fn(z, 2), fn(zs, 2)
                        floor = topk_instability(a1, a2, k, cfg.tie_abs, cfg.tie_rel)
                        shift = topk_instability(a1, a3, k, cfg.tie_abs, cfg.tie_rel)
                        rows.append(dict(dataset=name, seed=seed, model=kind, explainer=ex_name, budget=str(bud),
                                         record=int(split.idx_test[r_i]), floor=floor, shift=shift, net=shift - floor))
    sfx = out / "source"
    sfx.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(sfx / f"stochastic_{name}.csv", index=False)
