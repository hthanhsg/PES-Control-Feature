"""Classifiers of Sec. 5.1 plus the gradient-boosted detector requested in R2-M5."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_model(kind: str, seed: int, url: bool = False):
    if kind == "lr":
        return LogisticRegression(max_iter=800 if url else 1500, class_weight="balanced")
    if kind == "rf":
        return RandomForestClassifier(n_estimators=60, min_samples_leaf=2, class_weight="balanced_subsample",
                                      random_state=seed, n_jobs=-1)
    if kind == "gb":
        # Deterministic, TreeSHAP-compatible gradient boosting (R2-M5).
        return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=seed)
    raise ValueError(kind)


def proba(model, X: np.ndarray) -> np.ndarray:
    """Threat probability p(x) = P(y=1|x)."""
    return model.predict_proba(np.atleast_2d(X))[:, 1]


def score(model, X: np.ndarray, cls: np.ndarray | int, space: str = "prob") -> np.ndarray:
    """Score of the explained class: probability or logit of class `cls` (R1-M5, R3-M3)."""
    p = proba(model, X)
    pc = np.where(np.asarray(cls) == 1, p, 1.0 - p)
    if space == "prob":
        return pc
    pc = np.clip(pc, 1e-12, 1 - 1e-12)
    return np.log(pc) - np.log1p(-pc)
