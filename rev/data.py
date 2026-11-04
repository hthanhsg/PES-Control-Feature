"""Dataset loading and the source-study preprocessing of Sec. 5.1.

Loading order for each source dataset:
  1. <data_dir>/<name>.csv if present (recommended: use the same files as the archived run,
     so the SHA-256 values match the manuscript's provenance record);
  2. otherwise the UCI repository via `ucimlrepo` (ids 722, 327, 967).

CONFIRM the label mapping below against the archived pipeline: the class coded as threat
must be 1 after loading (malware / phishing).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

from .common import KEYWORDS, Config

LOG = logging.getLogger("rev")

SOURCES = {
    # name: (uci id, target column, value that denotes the threat class)
    "naticusdroid": (722, "Result", 1),      # CONFIRM: 1 = malware
    "uci_phishing": (327, "result", -1),     # CONFIRM: -1 = phishing in the UCI 327 coding
    "phiusiil": (967, "label", 0),           # UCI page: label 1 = legitimate, 0 = phishing
}
ID_LIKE = ("filename", "url", "domain", "tld", "title", "id")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(name: str, data_dir: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    uci_id, target, threat_value = SOURCES[name]
    csv = Path(data_dir) / f"{name}.csv"
    if csv.is_file():
        df = pd.read_csv(csv)
        LOG.info("%s: loaded %s (sha256 %s)", name, csv, sha256(csv)[:16])
    else:
        from ucimlrepo import fetch_ucirepo  # imported lazily: needs network
        ds = fetch_ucirepo(id=uci_id)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df.to_csv(csv, index=False) if Path(data_dir).is_dir() else None
        LOG.info("%s: fetched UCI id %d", name, uci_id)
    tcol = next((c for c in df.columns if c.lower() == target.lower()), None)
    if tcol is None:
        raise KeyError(f"{name}: target column '{target}' not found; columns: {list(df.columns)[:10]}...")
    y = (df[tcol] == threat_value).astype(int)
    X = df.drop(columns=[tcol])
    return X, y


@dataclass
class Split:
    dataset: str
    seed: int
    features: list[str]
    mi: np.ndarray               # mutual information of the retained features (training only)
    mi_rank: np.ndarray          # 0 = highest MI
    eligible: np.ndarray         # bool, keyword rule (Sec. 4.1)
    X_train: np.ndarray          # standardized
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    idx_test: np.ndarray         # row ids in the capped source table (record identity across seeds)
    raw_train: pd.DataFrame      # imputed raw values (for discrete-code and addition shifts)
    raw_test: pd.DataFrame
    mean_: np.ndarray
    scale_: np.ndarray
    median_std: np.ndarray       # training median in standardized space (primary baseline b)
    mean_std: np.ndarray         # training mean in standardized space (= 0)
    binary: np.ndarray           # bool, feature takes only {0,1} in training data


def cap_table(X: pd.DataFrame, y: pd.Series, cap: int, cap_seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= cap:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    Xc, _, yc, _ = train_test_split(X, y, train_size=cap, stratify=y, random_state=cap_seed)
    return Xc.reset_index(drop=True), yc.reset_index(drop=True)


def numeric_candidates(X_train: pd.DataFrame) -> list[str]:
    out = []
    for c in X_train.columns:
        if any(c.lower() == s or c.lower().endswith(s) for s in ID_LIKE) and not pd.api.types.is_numeric_dtype(X_train[c]):
            continue
        if not pd.api.types.is_numeric_dtype(X_train[c]):
            continue
        col = X_train[c]
        if col.notna().mean() < 0.95 or col.nunique(dropna=True) < 2:
            continue
        out.append(c)
    return out


def make_split(name: str, X: pd.DataFrame, y: pd.Series, seed: int, cfg: Config) -> Split:
    idx = np.arange(len(X))
    i_tr, i_te = train_test_split(idx, test_size=cfg.test_size, stratify=y, random_state=seed)
    Xtr, Xte = X.iloc[i_tr], X.iloc[i_te]
    ytr, yte = y.iloc[i_tr].to_numpy(), y.iloc[i_te].to_numpy()
    cand = numeric_candidates(Xtr)
    med = Xtr[cand].median()
    Xtr_i = Xtr[cand].fillna(med)
    mi = mutual_info_classif(Xtr_i.to_numpy(dtype=float), ytr, random_state=seed)
    order = np.argsort(-mi, kind="stable")[: cfg.max_features]
    feats = [cand[i] for i in order]
    mi_sel = mi[order]
    raw_tr = Xtr[feats].fillna(med[feats]).astype(float)
    raw_te = Xte[feats].fillna(med[feats]).astype(float)
    mean_, scale_ = raw_tr.mean().to_numpy(), raw_tr.std(ddof=0).replace(0, 1).to_numpy()
    Ztr = (raw_tr.to_numpy() - mean_) / scale_
    Zte = (raw_te.to_numpy() - mean_) / scale_
    eligible = np.array([any(k in f.lower() for k in KEYWORDS) for f in feats])
    if not eligible.any():
        eligible[:] = True
    binary = np.array([set(np.unique(raw_tr[f])) <= {0.0, 1.0} for f in feats])
    return Split(dataset=name, seed=seed, features=feats, mi=mi_sel, mi_rank=np.arange(len(feats)),
                 eligible=eligible, X_train=Ztr, X_test=Zte, y_train=ytr, y_test=yte, idx_test=i_te,
                 raw_train=raw_tr, raw_test=raw_te, mean_=mean_, scale_=scale_,
                 median_std=np.median(Ztr, axis=0), mean_std=Ztr.mean(axis=0), binary=binary)


def stratified_subset(y: np.ndarray, n: int | None, seed: int) -> np.ndarray:
    """Indices of a label-stratified subset fixed from true labels before any explanation."""
    idx = np.arange(len(y))
    if n is None or n >= len(y):
        return idx
    sel, _ = train_test_split(idx, train_size=n, stratify=y, random_state=seed)
    return np.sort(sel)
