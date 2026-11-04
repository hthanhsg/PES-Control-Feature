"""Turn record-level results into the LaTeX tables, figures and numeric macros of the revision.

Every file name here is referenced by \\input{} / \\includegraphics{} in the revised manuscript.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DATASET_LABEL as DL, MODEL_LABEL as ML, Config, ci, cluster_bootstrap, fmt, seed_t_interval, \
    write_macros, write_table
from .metrics import attainable_values, chance_instability, k_of

LOG = logging.getLogger("rev")


def _load_records(out: Path) -> pd.DataFrame:
    parts = []
    for f in sorted((out / "source").glob("records_*")):
        parts.append(pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f))
    df = pd.concat(parts, ignore_index=True)
    df["delta"] = df["D_s"] - df["D_c"]
    df["q_s"] = (df["p_s"] - df["p0"]).abs()
    df["q_c"] = (df["p_c"] - df["p0"]).abs()
    df["gamma"] = df["q_s"] - df["q_c"]
    yh0, yhs, yhc = df["p0"] >= .5, df["p_s"] >= .5, df["p_c"] >= .5
    df["flip_s"], df["flip_c"] = (yh0 != yhs).astype(int), (yh0 != yhc).astype(int)
    df["C_s"] = ((yh0 == df["y"].astype(bool)) & (yhs == df["y"].astype(bool))).astype(int)
    df["C_c"] = ((yh0 == df["y"].astype(bool)) & (yhc == df["y"].astype(bool))).astype(int)
    return df


def _sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def _seed_ci(g: pd.DataFrame, col: str = "delta"):
    return seed_t_interval(g.groupby("seed")[col].mean().to_numpy())


def _order(df):
    out = []
    for ds in ("naticusdroid", "uci_phishing", "phiusiil"):
        for m in ("lr", "rf", "gb"):
            if ((df["dataset"] == ds) & (df["model"] == m)).any():
                out.append((ds, m))
    return out


PRIM = dict(explainer="ablation", baseline="median", space="prob")


def mixed_model(g: pd.DataFrame):
    """delta_i ~ gamma_i + (flip_s - flip_c) + C(seed, Sum) with a random intercept per record (R1-M2 ii)."""
    import statsmodels.formula.api as smf
    h = g.assign(dflip=g["flip_s"] - g["flip_c"]).dropna(subset=["delta", "gamma"])
    if h["record_uid"].nunique() < 5 or h["delta"].std() == 0:
        return dict(icpt=np.nan, lo=np.nan, hi=np.nan, slope=np.nan, r2=np.nan, n=len(h))
    try:
        res = smf.mixedlm("delta ~ gamma + dflip + C(seed, Sum)", h, groups=h["record_uid"]).fit(method="lbfgs", disp=False)
        icpt, (lo, hi) = res.params["Intercept"], res.conf_int().loc["Intercept"].tolist()
        slope = res.params["gamma"]
    except Exception:  # fall back to OLS with cluster-robust errors
        res = smf.ols("delta ~ gamma + dflip + C(seed, Sum)", h).fit(cov_type="cluster", cov_kwds={"groups": h["record_uid"]})
        icpt, (lo, hi) = res.params["Intercept"], res.conf_int().loc["Intercept"].tolist()
        slope = res.params["gamma"]
    r2 = smf.ols("delta ~ gamma + dflip", h).fit().rsquared
    return dict(icpt=icpt, lo=lo, hi=hi, slope=slope, r2=r2, n=len(h))


def build(out: Path, cfg: Config) -> None:
    tdir, fdir = out / "tables", out / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    macros: dict[str, str] = {}
    (tdir / "rev_macros.tex").unlink(missing_ok=True)
    df = _load_records(out)
    d0 = int(df["d"].mode().iat[0])
    k0 = k_of(d0, cfg.k_frac)
    macros["RevChanceDk"] = fmt(chance_instability(d0, k0))
    macros["RevKzero"], macros["RevDzero"] = str(k0), str(d0)
    macros["RevAttainable"] = ", ".join(fmt(v, 2) for v in attainable_values(k0))
    macros["RevNseeds"] = str(df["seed"].nunique())

    prim = _sel(df, family="mask", severity=2, **PRIM)
    # ------------------------------------------------------------------ control ladder (R1-M1, R1-M2)
    rows = []
    for ds, m in _order(prim):
        g = _sel(prim, dataset=ds, model=m)
        cells = [DL[ds], ML[m]]
        for c in ("redistribute", "dp_matched", "op_matched", "mi_matched"):
            h = g[(g.control == c) & (g.valid_c == 1)]
            cells.append(ci(*_seed_ci(h)) if len(h) else "--")
        mm = mixed_model(g[(g.control == "redistribute") & (g.valid_c == 1)])
        cells.append(ci(mm["icpt"], mm["lo"], mm["hi"]))
        rows.append(cells)
        key = f"{DL[ds].replace(' ', '')}{ML[m]}"
        for c, nm in (("redistribute", "Orig"), ("dp_matched", "Dp"), ("op_matched", "Op"), ("mi_matched", "Mi")):
            h = g[(g.control == c) & (g.valid_c == 1)]
            macros[f"Rev{nm}{key}"] = fmt(_seed_ci(h)[0], 3, True) if len(h) else "--"
            macros[f"Rev{nm}Valid{key}"] = fmt(float(g[g.control == c]["valid_c"].mean()) * 100, 1)
        macros[f"RevIcpt{key}"] = fmt(mm["icpt"], 3, True)
        macros[f"RevSlope{key}"] = fmt(mm["slope"], 2, True)
        macros[f"RevRsq{key}"] = fmt(mm["r2"], 2)
    # distortion summaries used to answer R3-M2 quantitatively
    lad = {}
    for ds, m in _order(prim):
        g = _sel(prim, dataset=ds, model=m)
        lad[(ds, m)] = {c: _seed_ci(g[(g.control == c) & (g.valid_c == 1)])[0] for c in
                        ("redistribute", "dp_matched", "op_matched", "mi_matched")}
    diffs = lambda c: [abs(v["redistribute"] - v[c]) for v in lad.values() if np.isfinite(v[c])]
    macros["RevMaxGapDp"] = fmt(max(diffs("dp_matched"), default=np.nan))
    macros["RevMaxGapOp"] = fmt(max(diffs("op_matched"), default=np.nan))
    macros["RevMaxGapMi"] = fmt(max(diffs("mi_matched"), default=np.nan))
    orig = [v["redistribute"] for v in lad.values()]
    macros["RevPooledOrig"] = fmt(float(np.mean(orig)), 3, True)
    macros["RevMinOrig"], macros["RevMaxOrig"] = fmt(min(orig), 3, True), fmt(max(orig), 3, True)
    opv = [v["op_matched"] for v in lad.values() if np.isfinite(v["op_matched"])]
    macros["RevMinOp"], macros["RevMaxOp"] = fmt(min(opv), 3, True), fmt(max(opv), 3, True)
    write_table(tdir / "tab_ladder.tex",
                "Control ladder for the severity-two keyword-eligible MI-ranked mask (single-coordinate ablation, "
                "median baseline, probability scale). Each entry is the mean security-minus-control top-$k$ contrast "
                "$\\bar\\delta$ with a descriptive 95\\% Student-$t$ interval over split-seed means. Redistribution is the "
                "original matched control (Redistr.); the probability-matched control ($|\\Delta p|$) additionally matches $|\\Delta p|$ "
                "(tolerance 0.02); operation- and MI-rank-matched controls apply the same shrinkage towards the median "
                "to random (Operation) or rank-adjacent (MI rank) coordinates. The last column is the intercept of the "
                "record-level mixed model, i.e.\\ the contrast at $\\gamma=0$ without differential label flips. All "
                "held-out records of every split are included.",
                "tab:ladder", ["Dataset", "Model", "Redistr.", "$|\\Delta p|$", "Operation", "MI rank", "$\\gamma=0$"], rows,
                align="llrrrrr", size="\\scriptsize")

    # ------------------------------------------------------------------ mixed model details
    rows = []
    for ds, m in _order(prim):
        g = _sel(prim, dataset=ds, model=m, control="redistribute")
        g = g[g.valid_c == 1]
        mm = mixed_model(g)
        cb = cluster_bootstrap(g, "delta", B=cfg.bootstrap)
        rows.append([DL[ds], ML[m], fmt(mm["n"]), ci(*cb), fmt(mm["slope"], 2, True), ci(mm["icpt"], mm["lo"], mm["hi"]),
                     fmt(mm["r2"], 2)])
    write_table(tdir / "tab_mixed.tex",
                "Hierarchical uncertainty and record-level model (severity-two mask, primary explainer, redistribution "
                "control). The bootstrap interval resamples split seeds and then records within seeds. The model is "
                "$\\delta_i \\sim \\gamma_i + (F_i(T_s)-F_i(T_c)) + \\text{seed}$ with a random intercept per source record. "
                "$R^2$ is the share of record-level variance in $\\delta_i$ explained by probability response and "
                "differential label flips.", "tab:mixed",
                ["Dataset", "Model", "Pairs", "$\\bar\\delta$ (bootstrap)", "Slope $\\gamma$", "$\\gamma=0$", "$R^2$"],
                rows, align="llrrrrr", size="\\scriptsize")

    # ------------------------------------------------------------------ explainer / reference / space (R1-M1 iii, R1-M5 iii, R3-M3)
    cfgs = [("ablation", "median", "prob", "Med.\\,$p$"), ("ablation", "median", "logit", "Med.\\,logit"),
            ("ablation", "mean", "prob", "Mean\\,$p$"), ("ablation", "background", "prob", "Backgr.\\,$p$"),
            ("shap", "background", "native", "SHAP")]
    rows = []
    for ds, m in _order(prim):
        cells = [DL[ds], ML[m]]
        for e, b, s, _ in cfgs:
            g = _sel(df, family="mask", severity=2, explainer=e, baseline=b, space=s, dataset=ds, model=m, control="op_matched")
            g = g[g.valid_c == 1]
            cells.append(ci(*_seed_ci(g)) if len(g) else "--")
        rows.append(cells)
    write_table(tdir / "tab_explainers.tex",
                "Severity-two contrast against the operation-matched control under alternative attribution references "
                "and score scales for single-coordinate ablation (Med.: training median; Mean: training mean; Backgr.: "
                "interventional average over 100 training records; $p$: probability; logit: log-odds) and for "
                "model-specific Shapley explainers (SHAP: LinearSHAP for LR in logit space, interventional TreeSHAP for "
                "RF and GB on the probability scale).",
                "tab:explainers", ["Dataset", "Model"] + [c[3] for c in cfgs], rows, align="llrrrrr", size="\\scriptsize")

    # ------------------------------------------------------------------ signed agreement (R3-M3 c)
    rows = []
    for ds, m in _order(prim):
        g = _sel(prim, dataset=ds, model=m, control="op_matched")
        g = g[g.valid_c == 1]
        rows.append([DL[ds], ML[m], fmt(g["cos_s"].mean()), fmt(g["cos_c"].mean()), fmt(g["sf_s"].mean()), fmt(g["sf_c"].mean())])
    write_table(tdir / "tab_signed.tex",
                "Signed agreement between clean and transformed attributions (severity-two mask versus operation-matched "
                "control, primary explainer): mean signed cosine (Eq.~\\eqref{eq:cosine}) and the weighted share of clean "
                "top-$k$ features whose attribution sign reverses.", "tab:signed",
                ["Dataset", "Model", "Cosine, mask", "Cosine, control", "Sign flips, mask", "Sign flips, control"], rows)

    # ------------------------------------------------------------------ capability-defined shift (R2-M1)
    cap = _sel(df, family="add_perm", control="redistribute", **PRIM)
    if len(cap):
        rows = []
        for m in ("lr", "rf", "gb"):
            for sev, n_add in ((1, 1), (2, 2), (3, 4)):
                g = _sel(cap, model=m, severity=sev)
                if not len(g):
                    continue
                rows.append([ML[m], str(n_add), ci(*_seed_ci(g)), fmt(g["gamma"].mean(), 3, True),
                             fmt(g["flip_s"].mean()), fmt(g["flip_c"].mean()),
                             fmt(g.loc[g.y == 1, "flip_s"].mean())])
        write_table(tdir / "tab_capability.tex",
                    "Capability-defined Android shift on NATICUSdroid: adding the absent permissions most associated with "
                    "the benign class versus adding the same number of randomly chosen absent permissions. Flip$_a$ and "
                    "Flip$_r$ are label-change rates relative to the clean prediction for the two additions; Evasion is the "
                    "flip rate among malware records under the benign-associated addition. Each row covers all paired records "
                    "of all splits.",
                    "tab:capability", ["Model", "Added", "$\\bar\\delta$", "$\\bar\\gamma$",
                                       "Flip$_a$", "Flip$_r$", "Evasion"], rows, align="lrrrrrr", size="\\scriptsize")

    # ------------------------------------------------------------------ feature-domain-valid mask
    disc = _sel(df, family="mask_discrete", control="redistribute", **PRIM)
    if len(disc):
        rows = []
        for ds, m in _order(disc):
            g = _sel(disc, dataset=ds, model=m)
            rows.append([DL[ds], ML[m], fmt(float((g["n0_s"] == 0).mean())), ci(*_seed_ci(g[g.valid_c == 1])),
                         fmt(g["gamma"].mean(), 3, True)])
        write_table(tdir / "tab_discrete.tex",
                    "Severity-two mask with values snapped to observed training codes (feature-domain valid) against its "
                    "own matched redistribution control. The first numeric column is the share of records for which the "
                    "snapped transform leaves the vector unchanged.", "tab:discrete",
                    ["Dataset", "Model", "Unchanged", "$\\bar\\delta$ [seed CI]", "$\\bar\\gamma$"], rows)

    # ------------------------------------------------------------------ all severities and baselines (R1-M6)
    rows = []
    for ds, m in _order(df):
        for sev in (1, 2, 3):
            cells = [DL[ds], ML[m], str(sev)]
            for base in ("median", "mean"):
                for c in ("redistribute", "op_matched"):
                    g = _sel(df, family="mask", severity=sev, explainer="ablation", baseline=base, space="prob",
                             dataset=ds, model=m, control=c)
                    g = g[g.valid_c == 1]
                    cells.append(fmt(_seed_ci(g)[0], 3, True) if len(g) else "--")
            g = _sel(df, family="mask", severity=sev, dataset=ds, model=m, control="redistribute", **PRIM)
            cells.append(fmt(g["gamma"].mean(), 3, True))
            rows.append(cells)
    write_table(tdir / "tab_severity_full.tex",
                "All severities and both feature-removal references (single-coordinate ablation, probability scale): "
                "mean contrast against the redistribution (Red.) and operation-matched (Op.) controls, and the mean "
                "excess probability response $\\bar\\gamma$ (median reference, redistribution control).",
                "tab:severityfull", ["Dataset", "Model", "Sev.", "Med./Red.", "Med./Op.", "Mean/Red.", "Mean/Op.",
                                     "$\\bar\\gamma$"], rows, align="llrrrrrr", size="\\scriptsize")

    # ------------------------------------------------------------------ k sensitivity, chance level, grouped (R1-M5 iv, R2-M5)
    kcols = sorted(c for c in df.columns if c.startswith("D_s_k"))
    rows = []
    for ds, m in _order(prim):
        g = _sel(prim, dataset=ds, model=m, control="op_matched")
        g = g[g.valid_c == 1]
        cells = [DL[ds], ML[m]]
        for c in kcols:
            cc = c.replace("D_s", "D_c")
            cells.append(fmt((g[c] - g[cc]).groupby(g["seed"]).mean().mean(), 3, True))
        cells.append(fmt((g["D_s_group"] - g["D_c_group"]).groupby(g["seed"]).mean().mean(), 3, True))
        rows.append(cells)
    hdr = [f"$k={c[5:]}\\%$" for c in kcols]
    chance = "; ".join(f"$k={c[5:]}\\%$: {fmt(chance_instability(d0, k_of(d0, int(c[5:]) / 100)), 2)}" for c in kcols)
    write_table(tdir / "tab_ksens.tex",
                f"Sensitivity of the severity-two contrast (operation-matched control) to the top-$k$ fraction and to "
                f"grouping correlated features ($|\\rho_S|\\geq0.8$) before ranking. Chance-level $D_k$ for $d={d0}$: {chance}.",
                "tab:ksens", ["Dataset", "Model"] + hdr + ["Grouped"], rows)

    # ------------------------------------------------------------------ ties, saturation, zero movement (R1-M5 i, ii)
    clean = pd.concat([pd.read_csv(f) for f in sorted((out / "source").glob("clean_*.csv"))], ignore_index=True)
    rows = []
    for ds, m in _order(prim):
        g = _sel(prim, dataset=ds, model=m, control="redistribute")
        z = g[g["n0_s"] == 0]
        gl = _sel(df, family="mask", severity=2, explainer="ablation", baseline="median", space="logit", dataset=ds,
                  model=m, control="redistribute")
        sat = clean[(clean.dataset == ds) & (clean.model == m)]["saturated"].mean()
        rows.append([DL[ds], ML[m], fmt(g["tie_clean"].mean()), fmt(gl["tie_clean"].mean()) if len(gl) else "--",
                     fmt(sat), f"{int((z['D_s'] > 0).sum())}/{len(z)}"])
    write_table(tdir / "tab_ties.tex",
                "Metric-degeneracy diagnostics: share of clean explanations whose $k$-th and $(k{+}1)$-th magnitudes are "
                "tied under the harmonized tolerance, on the probability and logit scales; share of explained records "
                "with clean probability outside $[10^{-3},1-10^{-3}]$; and records with zero realized movement that "
                "nevertheless show $D_k>0$.", "tab:ties",
                ["Dataset", "Model", "Ties, $p$", "Ties, logit", "Saturated", "Zero-move, $D_k>0$"], rows)

    # ------------------------------------------------------------------ clean performance (R1 minor, R2 minor)
    rows = []
    for ds, m in _order(clean):
        g = clean[(clean.dataset == ds) & (clean.model == m)]
        rows.append([DL[ds], ML[m], fmt(g.prevalence.mean()), fmt(g.ba.mean(), 4), fmt(g.f1.mean(), 4), fmt(g.auc.mean(), 4),
                     fmt(g.precision.mean(), 4), fmt(g.recall.mean(), 4), fmt(g.fpr.mean(), 4), fmt(g.cohort_acc.mean(), 4)])
    write_table(tdir / "tab_clean.tex",
                f"Clean source-model performance on the full held-out partitions, averaged over {clean.seed.nunique()} "
                "splits, with threat prevalence after capping, false-positive rate at the 0.5 threshold, and accuracy on "
                "the explained cohort.", "tab:clean",
                ["Dataset", "Model", "Prev.", "BA", "F1", "AUC", "Prec.", "Rec.", "FPR", "Cohort acc."], rows,
                align="llrrrrrrrr", size="\\scriptsize")

    # ------------------------------------------------------------------ eligibility and redundancy (R1-M1 iv, R2-M5)
    el = pd.concat([pd.read_csv(f) for f in sorted((out / "source").glob("eligibility_*.csv"))], ignore_index=True)
    rd = pd.concat([pd.read_csv(f) for f in sorted((out / "source").glob("redundancy_*.csv"))], ignore_index=True)
    rows = []
    for ds in ("naticusdroid", "uci_phishing", "phiusiil"):
        g = el[(el.dataset == ds) & (el.severity == 2)]
        r = rd[rd.dataset == ds]
        if not len(g):
            continue
        first = g.iloc[0]
        rows.append([DL[ds], f"{g.n_eligible.mean():.1f}/{int(first.d)}", str(int(first.n_masked)),
                     ", ".join("\\texttt{" + f.split(".")[-1].replace("_", "\\_") + "}" for f in first.masked.split(";")),
                     f"{r.n_clusters.mean():.1f}", fmt(r.max_single_auc.mean()), f"{r.n_single_auc_095.mean():.1f}"])
    write_table(tdir / "tab_eligible.tex",
                "Keyword eligibility and redundancy. Eligible features (mean over splits) among the $d$ retained, "
                "features masked at severity two in the first split, number of correlation clusters "
                "($|\\rho_S|\\ge0.8$), best single-feature training AUC, and number of features with single-feature "
                "AUC $\\ge 0.95$.", "tab:eligible",
                ["Dataset", "Elig.", "Mask.", "Masked features (sev.\\ 2, first split)", "Clust.",
                 "AUC$_1$", "$\\ge0.95$"], rows, align="lrrp{0.36\\linewidth}rrr", size="\\scriptsize")

    # ------------------------------------------------------------------ case study (R2-M6)
    cs = [pd.read_csv(f) for f in sorted((out / "source").glob("cases_*.csv")) if f.stat().st_size > 5]
    if cs:
        cs = pd.concat(cs, ignore_index=True)
        rows = []
        for (ds, rec), g in cs.groupby(["dataset", "record_uid"], sort=False):
            for _, r in g.iterrows():
                tops = r.top.split(";")
                rows.append([DL[ds] if r.input == "clean" else "", str(rec) if r.input == "clean" else "",
                             "malware/phish" if r.y == 1 else "benign", r.input, fmt(r.p),
                             "; ".join("\\texttt{" + t.split("=")[0].split(".")[-1].replace("_", "\\_\\allowbreak{}") + "}" for t in tops),
                             fmt(r.D) if r.input != "clean" else ""])
        write_table(tdir / "tab_case.tex",
                    "Analyst-facing view for two records per dataset (first split, RF, severity-two mask): the top-$k$ "
                    "features shown before and after the transform, the threat probability and the record's $D_k$.",
                    "tab:case", ["Dataset", "Record", "Label", "Input", "$p$", "Top-$k$ features (by $|a_j|$)", "$D_k$"], rows,
                    align="llllr>{\\raggedright\\arraybackslash}p{0.30\\linewidth}r", size="\\scriptsize")

    # ------------------------------------------------------------------ stochastic floors (R1-M3)
    st = [pd.read_csv(f) for f in sorted((out / "source").glob("stochastic_*.csv"))]
    if st:
        st = pd.concat(st, ignore_index=True)
        rows = []
        for (ds, ex), g in st.groupby(["dataset", "explainer"], sort=False):
            cells = [DL[ds], ex]
            for bud in ("128", "512", "2048", "default"):
                h = g[g.budget == bud]
                cells.append(fmt(h.floor.mean()) if len(h) else "--")
            cells += [fmt(int(g.groupby("budget").size().max())), fmt(float((g[g.budget == "default"].net < 0).mean()))]
            rows.append(cells)
        write_table(tdir / "tab_stochastic.tex",
                    "Identity floor (mean $D_k$ between two independent runs on the same input) for LIME and KernelSHAP by "
                    "sampling budget; ``default'' is 5{,}000 samples for LIME and $2d+2048$ for KernelSHAP. The last "
                    "column is the share of records whose shifted-input instability is below the identity floor at the "
                    "default budget.", "tab:stochastic",
                    ["Dataset", "Explainer", "128", "512", "2,048", "Default", "Cases", "Net $<0$"], rows, align="llrrrrrr")
        macros["RevStochCases"] = fmt(int(st.groupby(["dataset", "explainer", "budget"]).size().median()))
        for ex in ("LIME", "KernelSHAP"):
            for bud in ("128", "default"):
                h = st[(st.explainer == ex) & (st.budget == bud)]
                tag = ex.replace("KernelSHAP", "KS") + ("Def" if bud == "default" else "Low")   # macro names cannot contain digits
                if len(h):
                    per = h.groupby("dataset")["floor"].mean()
                    macros[f"RevFloor{tag}Min"], macros[f"RevFloor{tag}Max"] = fmt(per.min()), fmt(per.max())
        h = st[st.budget == "default"]
        macros["RevNetNegDef"] = fmt(float((h.net < 0).mean()) * 100, 1) if len(h) else "--"

    # ------------------------------------------------------------------ figures
    _figures(df, fdir)

    # ------------------------------------------------------------------ URL study
    _url_tables(out, tdir, cfg, macros)
    write_macros(tdir / "rev_macros.tex", macros)
    LOG.info("tables written to %s", tdir)


def _figures(df: pd.DataFrame, fdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    prim = _sel(df, family="mask", control="redistribute", **PRIM)
    models = [m for m in ("lr", "rf", "gb") if (prim.model == m).any()]
    colors = {"naticusdroid": "#1f5582", "uci_phishing": "#c0612b", "phiusiil": "#3a6f5c"}
    marks = {"naticusdroid": "o", "uci_phishing": "s", "phiusiil": "^"}
    # Cell-level coupling (replaces Fig. 2, full axis label)
    fig, axes = plt.subplots(1, len(models), figsize=(2.4 * len(models) + 0.6, 2.4), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        for ds in colors:
            for sev in (1, 2, 3):
                g = _sel(prim, model=m, dataset=ds, severity=sev)
                if len(g):
                    gx = g.groupby("seed")["gamma"].mean().mean()
                    gy = g.groupby("seed")["delta"].mean().mean()
                    ax.scatter(gx, gy, marker=marks[ds], facecolors="none", edgecolors=colors[ds], s=28,
                               label=DL[ds] if sev == 1 else None)
                    ax.annotate(str(sev), (gx, gy), xytext=(3, 2), textcoords="offset points", fontsize=6, color=colors[ds])
        ax.axhline(0, ls="--", lw=0.6, color="grey")
        ax.axvline(0, lw=0.6, color="grey")
        ax.set_title(ML[m])
        ax.set_xlabel(r"Excess $|\Delta p|$, $\bar\gamma$")
    np.atleast_1d(axes)[0].set_ylabel(r"Contrast $\bar\delta$ (mask $-$ control)")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(fdir / "fig_coupling.pdf")
    plt.close(fig)
    # Record-level scatter (R1-M2 iii)
    s2 = prim[prim.severity == 2]
    fig, axes = plt.subplots(len(models), 3, figsize=(7.2, 2.0 * len(models)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, m in enumerate(models):
        for j, ds in enumerate(colors):
            ax = axes[i, j]
            g = _sel(s2, model=m, dataset=ds)
            ax.scatter(g["gamma"], g["delta"], s=2, alpha=0.25, color=colors[ds], rasterized=True)
            ax.axhline(0, ls="--", lw=0.5, color="grey")
            ax.axvline(0, lw=0.5, color="grey")
            if i == 0:
                ax.set_title(DL[ds])
            if j == 0:
                ax.set_ylabel(f"{ML[m]}: $\\delta_i$")
            if i == len(models) - 1:
                ax.set_xlabel(r"$\gamma_i$")
    fig.tight_layout()
    fig.savefig(fdir / "fig_record_scatter.pdf", dpi=200)
    plt.close(fig)
    # Forest plot of the control ladder (replaces Fig. 4)
    lad = _sel(df, family="mask", severity=2, **PRIM)
    combos = _order(lad)
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(combos) + 0.9))
    styles = {"redistribute": ("o", "#1f5582"), "dp_matched": ("D", "#7a4fa3"), "op_matched": ("s", "#c0612b")}
    for yi, (ds, m) in enumerate(combos):
        for off, (c, (mk, col)) in zip((-0.2, 0, 0.2), styles.items()):
            g = _sel(lad, dataset=ds, model=m, control=c)
            g = g[g.valid_c == 1]
            mu, lo, hi = _seed_ci(g)
            ax.errorbar(mu, yi + off, xerr=[[mu - lo], [hi - mu]], fmt=mk, color=col, ms=3.5, lw=0.9,
                        label={"redistribute": "Redistribution", "dp_matched": r"$|\Delta p|$-matched",
                               "op_matched": "Operation-matched"}[c] if yi == 0 else None)
    ax.axvline(0, ls="--", lw=0.6, color="grey")
    ax.set_yticks(range(len(combos)), [f"{DL[d]} / {ML[m]}" for d, m in combos])
    ax.invert_yaxis()
    ax.set_xlabel(r"Severity-two contrast $\bar\delta$ (95% $t$ interval over split seeds)")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(fdir / "fig_forest.pdf")
    plt.close(fig)


def _url_tables(out: Path, tdir: Path, cfg: Config, macros: dict) -> None:
    ud = out / "url"
    if not (ud / "performance.csv").is_file():
        return
    perf = pd.read_csv(ud / "performance.csv")
    intg = pd.read_csv(ud / "in_target.csv") if (ud / "in_target.csv").stat().st_size > 5 else pd.DataFrame()
    allp = pd.concat([perf, intg], ignore_index=True)
    rows = []
    for m in ("lr", "rf"):
        for pop in allp.population.unique():
            g = allp[(allp.model == m) & (allp.population == pop)]
            if not len(g):
                continue
            short = {"PhiUSIIL holdout": "Source holdout", "PhishVN grouped": "VN in-target",
                     "PhishVN temporal": "VN temporal"}.get(pop, pop.replace("PhishVN", "VN"))
            rows.append([ML[m], short, fmt(int(g.n.iloc[0])), fmt(g.ba.mean()), fmt(g.f1.mean()), fmt(g.auc.mean()),
                         fmt(g.pr_auc.mean()), fmt(g.brier.mean()), fmt(g.ece.mean()), fmt(g.fpr.mean()),
                         fmt(g.pred_pos_rate.mean())])
    write_table(tdir / "tab_external.tex",
                "URL study under the shared 17-feature lexical extractor. Cross-source rows use models trained on "
                "PhiUSIIL (domain-grouped splits); in-target rows train and test within PhishVN (domain-grouped and, "
                "where timestamps allow, temporal splits) and give the attainable reference for this representation. "
                "Means over split seeds; threshold 0.5. VN: PhishVN.", "tab:external",
                ["Model", "Population", "$n$", "BA", "F1", "AUC", "PR-AUC", "Brier", "ECE", "FPR", "Pred.\\ pos."], rows,
                align="llrrrrrrrrr", size="\\scriptsize")
    for m in ("lr", "rf"):
        g = perf[(perf.model == m) & (perf.population == "PhishVN all")]
        if len(g):
            macros[f"RevBenignHigh{ML[m]}"] = fmt(g.benign_p_gt_09.mean())
            macros[f"RevPredPos{ML[m]}"] = fmt(g.pred_pos_rate.mean())
        h = intg[(intg.model == m) & (intg.population == "PhishVN grouped")] if len(intg) else pd.DataFrame()
        if len(h):
            macros[f"RevInTargetAUC{ML[m]}"] = fmt(h.auc.mean())
            macros[f"RevInTargetBA{ML[m]}"] = fmt(h.ba.mean())
    fr = pd.read_csv(ud / "fragment.csv")
    fr = fr[fr.target == "predicted"]
    cols = []
    for m in ("lr", "rf"):
        g = fr[fr.model == m]
        g = g.assign(delta=g.D_frag - g.D_ctrl)
        mu, lo, hi = seed_t_interval(g.groupby("seed")["delta"].mean())
        cols.append(dict(rows=fmt(int(len(g))), ok=fmt(g.request_target_ok.mean()), df=fmt(g.D_frag.mean()), dc=fmt(g.D_ctrl.mean()),
                         d=ci(mu, lo, hi), flips=fmt(int(g.flip.sum())),
                         ev=fmt(int(((g.direction == "threat->benign") & (g.y == 1)).sum())),
                         fa=fmt(int(((g.direction == "benign->threat") & (g.y == 0)).sum())),
                         qf=fmt((g.p_frag - g.p0).abs().mean()), qc=fmt((g.p_ctrl - g.p0).abs().mean()),
                         tf=fmt(g.train_frac_fragment.mean()), z=fmt(g.z_fragment_len.mean(), 1)))
        macros[f"RevFragEvasion{ML[m]}"] = cols[-1]["ev"]
        macros[f"RevFragFalseAlarm{ML[m]}"] = cols[-1]["fa"]
        macros[f"RevFragZ{ML[m]}"] = cols[-1]["z"]
        macros["RevFragTrainFrac"] = cols[-1]["tf"]
    labels = [("rows", "Paired URL rows"), ("ok", "Request target preserved"), ("df", "Fragment top-$k$ instability"),
              ("dc", "Matched-control top-$k$ instability"), ("d", "Fragment $-$ control [seed CI]"),
              ("flips", "Label flips"), ("ev", "\\quad phishing $\\to$ benign (evasion)"),
              ("fa", "\\quad benign $\\to$ phishing (false alarm)"), ("qf", "Mean $|\\Delta p|$, fragment"),
              ("qc", "Mean $|\\Delta p|$, control"), ("tf", "Training URLs with a fragment"),
              ("z", "Standardized fragment length after edit")]
    write_table(tdir / "tab_fragment.tex",
                "Paired URL-fragment edits on source holdout URLs (explained class: clean prediction). Edits append "
                "\\texttt{\\#section-} followed by eight random alphanumeric characters, replacing any existing fragment.",
                "tab:url", ["Diagnostic", "LR", "RF"], [[lab, cols[0][k], cols[1][k]] for k, lab in labels], align="lrr")
    ad = pd.read_csv(ud / "attribution_distance.csv") if (ud / "attribution_distance.csv").stat().st_size > 5 else pd.DataFrame()
    if len(ad):
        rows = []
        for (m, c, n), g in ad.groupby(["model", "cls", "n_per"]):
            rows.append([ML[m], "phishing" if c == 1 else "benign", str(n), fmt(g.within.mean()), fmt(g.across.mean()),
                         f"[{fmt(g.across_lo.mean())}, {fmt(g.across_hi.mean())}]"])
        write_table(tdir / "tab_attrdist.tex",
                    "Unpaired class-conditioned $\\ell_1$ distances between mean normalized absolute attribution vectors: "
                    "within-source reference versus source-to-PhishVN, by sample size per class and domain, with "
                    "descriptive record-bootstrap intervals for the cross-source distance.", "tab:attrdist",
                    ["Model", "Class", "$n$ per group", "Within source", "Across sources", "Across, 95\\% interval"], rows)
    fd = pd.read_csv(ud / "feature_distributions.csv")
    rows = []
    for f in ("url_len", "path_len", "query_len", "fragment_len", "n_slash", "n_subdomain"):
        cells = [f.replace("_", "\\_")]
        for src in ("PhiUSIIL", "PhishVN"):
            for c in (0, 1):
                g = fd[(fd.source == src) & (fd.cls == c) & (fd.feature == f)]
                cells.append(fmt(g["mean"].iloc[0], 1) if len(g) else "--")
        for c in (0, 1):
            g = fd[(fd.source.str.startswith("KS")) & (fd.cls == c) & (fd.feature == f)]
            cells.append(fmt(g["mean"].iloc[0], 2) if len(g) else "--")
        rows.append(cells)
    write_table(tdir / "tab_featdist.tex",
                "Selected lexical features by source (Phi.: PhiUSIIL; VN: PhishVN) and class (B: benign; P: phishing), "
                "means, with two-sample Kolmogorov--Smirnov statistics between sources within each class.", "tab:featdist",
                ["Feature", "Phi.\\ B", "Phi.\\ P", "VN B", "VN P", "KS B", "KS P"], rows, align="lrrrrrr")
