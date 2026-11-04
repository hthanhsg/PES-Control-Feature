"""Attribution methods. All return arrays of shape (n, d) for the explained class `cls`."""
from __future__ import annotations

import logging
import warnings

import numpy as np

from .models import proba, score

LOG = logging.getLogger("rev")


def ablation(model, Z: np.ndarray, b: np.ndarray, cls: np.ndarray, space: str = "prob",
             chunk: int = 20000) -> np.ndarray:
    """Single-coordinate ablation a_j = s(z) - s(z with z_j := b_j) (Sec. 5.1)."""
    Z = np.atleast_2d(Z)
    n, d = Z.shape
    cls = np.broadcast_to(np.asarray(cls), (n,))
    base = score(model, Z, cls, space)
    out = np.empty((n, d))
    per = max(1, chunk // d)
    for s in range(0, n, per):
        Zs = Z[s:s + per]
        m = len(Zs)
        R = np.repeat(Zs, d, axis=0)
        cols = np.tile(np.arange(d), m)
        R[np.arange(m * d), cols] = b[cols]
        sc = score(model, R, np.repeat(cls[s:s + per], d), space).reshape(m, d)
        out[s:s + per] = base[s:s + per, None] - sc
    return out


def ablation_background(model, Z: np.ndarray, B: np.ndarray, cls: np.ndarray, space: str = "prob") -> np.ndarray:
    """Interventional ablation: a_j = s(z) - E_{z' in B} s(z with z_j := z'_j)  (R1-M1 iii)."""
    Z = np.atleast_2d(Z)
    n, d = Z.shape
    cls = np.broadcast_to(np.asarray(cls), (n,))
    base = score(model, Z, cls, space)
    out = np.empty((n, d))
    K = len(B)
    for i in range(n):
        R = np.repeat(Z[i:i + 1], d * K, axis=0)
        rows = np.arange(d * K)
        cols = np.repeat(np.arange(d), K)
        R[rows, cols] = B[np.tile(np.arange(K), d), cols]
        sc = score(model, R, np.full(d * K, cls[i]), space).reshape(d, K).mean(axis=1)
        out[i] = base[i] - sc
    return out


class ShapExplainer:
    """LinearSHAP (logistic regression, logit space) or interventional TreeSHAP (tree models)."""

    def __init__(self, model, kind: str, background: np.ndarray):
        import shap
        self.kind = kind
        if kind == "lr":
            self.ex = shap.LinearExplainer(model, background)
            self.name = "LinearSHAP"
        else:
            self.ex = shap.TreeExplainer(model, data=background, feature_perturbation="interventional",
                                         model_output="probability")
            self.name = "TreeSHAP"

    def __call__(self, Z: np.ndarray, cls: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = self.ex.shap_values(np.atleast_2d(Z))
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 1]
        sign = np.where(np.asarray(cls) == 1, 1.0, -1.0)[:, None]
        return sv * sign


def lime_attr(model, z: np.ndarray, train: np.ndarray, cls: int, budget, seed: int, features=None) -> np.ndarray:
    from lime.lime_tabular import LimeTabularExplainer
    n_samples = 5000 if budget == "default" else int(budget)   # lime package default
    ex = LimeTabularExplainer(train, mode="classification", discretize_continuous=False,
                              feature_names=features, random_state=seed)
    e = ex.explain_instance(z, model.predict_proba, labels=(int(cls),), num_features=train.shape[1],
                            num_samples=n_samples)
    a = np.zeros(train.shape[1])
    for j, w in e.local_exp[int(cls)]:
        a[j] = w
    return a


def kernel_shap_attr(model, z: np.ndarray, bg, cls: int, budget, seed: int) -> np.ndarray:
    import shap
    f = lambda X: score(model, X, int(cls), "prob")
    np.random.seed(seed)
    ex = shap.KernelExplainer(f, bg)
    ns = "auto" if budget == "default" else int(budget)              # auto = 2d + 2048
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = ex.shap_values(np.atleast_2d(z), nsamples=ns, silent=True)
    return np.asarray(v).reshape(-1)
