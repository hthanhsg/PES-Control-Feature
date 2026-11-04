"""Explanation-similarity metrics with one harmonized tie convention (R1-M5)."""
from __future__ import annotations

from math import comb

import numpy as np
from scipy.stats import spearmanr


def k_of(d: int, k_frac: float) -> int:
    return max(1, int(np.ceil(k_frac * d - 1e-12)))


def topk_weights(a: np.ndarray, k: int, tie_abs: float, tie_rel: float) -> np.ndarray:
    """Fractional top-k membership (Eq. 9): ties at the cutoff share the remaining mass."""
    m = np.abs(np.asarray(a, dtype=float))
    d = len(m)
    if k >= d:
        return np.ones(d)
    tol = tie_abs + tie_rel * (m.max() if d else 0.0)
    v = np.sort(m)[::-1][k - 1]
    above = m > v + tol
    tied = np.abs(m - v) <= tol
    w = above.astype(float)
    rem = k - above.sum()
    if tied.sum() > 0:
        w[tied] = rem / tied.sum()
    return w


def topk_instability(a, a2, k, tie_abs, tie_rel) -> float:
    w1, w2 = topk_weights(a, k, tie_abs, tie_rel), topk_weights(a2, k, tie_abs, tie_rel)
    den = np.maximum(w1, w2).sum()
    return float(1.0 - np.minimum(w1, w2).sum() / den) if den > 0 else 0.0


def topk_instability_batch(A, A2, k, tie_abs, tie_rel) -> np.ndarray:
    return np.array([topk_instability(a, b, k, tie_abs, tie_rel) for a, b in zip(A, A2)])


def signed_cosine(a, a2, tol: float = 1e-10) -> float:
    n1, n2 = np.linalg.norm(a), np.linalg.norm(a2)
    if n1 < tol and n2 < tol:
        return 1.0
    if n1 < tol or n2 < tol:
        return 0.0
    return float(np.dot(a, a2) / (n1 * n2))


def spearman(a, a2) -> float:
    if np.array_equal(a, a2):
        return 1.0
    with np.errstate(all="ignore"):
        r = spearmanr(a, a2).statistic
    return float(r) if np.isfinite(r) else np.nan


def sign_flip_rate(a, a2, k, tie_abs, tie_rel) -> float:
    """Weighted share of the clean top-k features whose attribution sign reverses (R3-M3)."""
    w = topk_weights(a, k, tie_abs, tie_rel)
    flip = (np.sign(a) * np.sign(a2)) < 0
    return float((w * flip).sum() / w.sum()) if w.sum() > 0 else 0.0


def cutoff_tied(a, k, tie_abs, tie_rel) -> bool:
    """True if the k-th and (k+1)-th magnitudes are within the tie tolerance (R1-M5 ii)."""
    m = np.sort(np.abs(a))[::-1]
    if k >= len(m):
        return False
    tol = tie_abs + tie_rel * m[0]
    return bool(abs(m[k - 1] - m[k]) <= tol)


def chance_instability(d: int, k: int) -> float:
    """Expected D_k for two independent uniformly random top-k sets without ties (R1-M5 iv)."""
    tot = comb(d, k)
    e = 0.0
    for x in range(0, k + 1):
        p = comb(k, x) * comb(d - k, k - x) / tot
        e += p * (x / (2 * k - x))
    return 1.0 - e


def attainable_values(k: int) -> list[float]:
    return [1 - x / (2 * k - x) for x in range(k, -1, -1)]
