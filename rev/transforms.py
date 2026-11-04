"""Paired transformations and the control ladder requested in review.

All operations act on standardized feature vectors z (rows of Split.X_test).

  mask            keyword-eligible, MI-ranked shrinkage towards the reference b (Eq. 7)
  redistribute    original matched control: realized displacement magnitudes placed on random
                  coordinates with random signs (Eq. 8)                       [CONFIRM coordinate pool]
  op_matched      same shrink-towards-b operation on random coordinates, l0/l2 matched   (R1-M1 ii)
  mi_matched      same operation on non-selected coordinates with the nearest MI ranks (R1-M1 ii)
  dp_matched      redistribute control chosen among K candidates to match |delta p|     (R1-M2 i)
  mask_discrete   mask followed by mapping to observed training codes (feature-domain valid)
  add_perm        attacker-capability shift for binary permissions: add absent permissions (R2-M1)
  add_random      l0-matched control: add randomly chosen absent binary features        (R2-M1)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import MASK_FRACTIONS, MASK_STRENGTHS
from .data import Split


@dataclass
class TResult:
    z: np.ndarray            # transformed vector
    valid: bool
    n0: int                  # realized changed coordinates
    l2: float                # realized Euclidean displacement
    info: str = ""


def realized(delta: np.ndarray, eps: float) -> np.ndarray:
    return np.abs(delta) > eps


def mask_coords(split: Split, severity: int) -> np.ndarray:
    elig = np.flatnonzero(split.eligible)               # already in decreasing-MI order
    n_sel = max(1, int(round(MASK_FRACTIONS[severity - 1] * len(elig))))
    return elig[:n_sel]


def mask(z: np.ndarray, b: np.ndarray, J: np.ndarray, severity: int, eps: float) -> TResult:
    a = MASK_STRENGTHS[severity - 1]
    zt = z.copy()
    zt[J] = (1 - a) * z[J] + a * b[J]
    d = zt - z
    r = realized(d, eps)
    return TResult(zt, True, int(r.sum()), float(np.linalg.norm(d)))


def redistribute(z: np.ndarray, delta: np.ndarray, rng: np.random.Generator, eps: float,
                 pool: np.ndarray | None = None, l2_tol: float = 2e-5) -> TResult:
    r = realized(delta, eps)
    mags = np.abs(delta[r])
    n0 = len(mags)
    if n0 == 0:
        return TResult(z.copy(), True, 0, 0.0, "identity")
    pool = np.arange(len(z)) if pool is None else pool
    if len(pool) < n0:
        return TResult(z.copy(), False, 0, 0.0, "pool too small")
    pos = rng.choice(pool, size=n0, replace=False)
    zt = z.copy()
    zt[pos] = z[pos] + rng.permutation(mags) * rng.choice([-1.0, 1.0], size=n0)
    d = zt - z
    ok = realized(d, eps).sum() == n0 and abs(np.linalg.norm(d) - np.linalg.norm(delta)) <= l2_tol
    return TResult(zt, bool(ok), int(realized(d, eps).sum()), float(np.linalg.norm(d)))


def _shrink_to_l2(z, b, S, target_l2, eps):
    gap = b[S] - z[S]
    norm = np.linalg.norm(gap)
    if norm <= eps:
        return None
    alpha = target_l2 / norm
    if alpha > 1.0 + 1e-12:
        return None
    zt = z.copy()
    zt[S] = z[S] + alpha * gap
    return zt, alpha


def op_matched(z: np.ndarray, b: np.ndarray, n0: int, l2: float, rng: np.random.Generator, eps: float,
               exclude: np.ndarray | None = None, tries: int = 200) -> TResult:
    """Shrink n0 random coordinates towards b with a common strength chosen to match l2."""
    if n0 == 0:
        return TResult(z.copy(), True, 0, 0.0, "identity")
    movable = np.flatnonzero(np.abs(b - z) > eps)
    if exclude is not None:
        movable = np.setdiff1d(movable, exclude)
    if len(movable) < n0:
        return TResult(z.copy(), False, 0, 0.0, "too few movable coordinates")
    for _ in range(tries):
        S = rng.choice(movable, size=n0, replace=False)
        res = _shrink_to_l2(z, b, S, l2, eps)
        if res is not None:
            zt, alpha = res
            d = zt - z
            return TResult(zt, realized(d, eps).sum() == n0, int(realized(d, eps).sum()),
                           float(np.linalg.norm(d)), f"alpha={alpha:.3f}")
    return TResult(z.copy(), False, 0, 0.0, "l2 not attainable")


def mi_matched(z: np.ndarray, b: np.ndarray, J_real: np.ndarray, l2: float, mi_rank: np.ndarray,
               eps: float, J_all: np.ndarray) -> TResult:
    """Same operation on the unselected coordinates whose MI ranks are closest to the masked ones."""
    n0 = len(J_real)
    if n0 == 0:
        return TResult(z.copy(), True, 0, 0.0, "identity")
    movable = [j for j in np.flatnonzero(np.abs(b - z) > eps) if j not in set(J_all)]
    chosen: list[int] = []
    for j in J_real:
        cands = sorted((c for c in movable if c not in chosen), key=lambda c: (abs(mi_rank[c] - mi_rank[j]), c))
        if not cands:
            return TResult(z.copy(), False, 0, 0.0, "no rank-matched coordinate")
        chosen.append(cands[0])
    res = _shrink_to_l2(z, b, np.array(chosen), l2, eps)
    if res is None:
        return TResult(z.copy(), False, 0, 0.0, "l2 not attainable")
    zt, alpha = res
    d = zt - z
    return TResult(zt, realized(d, eps).sum() == n0, int(realized(d, eps).sum()), float(np.linalg.norm(d)),
                   f"alpha={alpha:.3f}")


def dp_matched(z: np.ndarray, delta: np.ndarray, target_q: float, p_clean: float, prob_fn, rng, eps: float,
               K: int, tol: float) -> tuple[TResult, float]:
    """Among K redistribute controls, keep the one whose |delta p| is closest to target_q (R1-M2)."""
    cands = [redistribute(z, delta, rng, eps) for _ in range(K)]
    cands = [c for c in cands if c.valid]
    if not cands:
        return TResult(z.copy(), False, 0, 0.0, "no valid candidate"), np.nan
    P = prob_fn(np.vstack([c.z for c in cands]))
    gaps = np.abs(np.abs(P - p_clean) - target_q)
    i = int(np.argmin(gaps))
    best = cands[i]
    best.valid = bool(gaps[i] <= tol)
    best.info = f"gap={gaps[i]:.4f}"
    return best, float(gaps[i])


def mask_discrete(split: Split, row: int, J: np.ndarray, severity: int, eps: float) -> TResult:
    """Mask in raw space, then snap each masked value to the nearest observed training code."""
    a = MASK_STRENGTHS[severity - 1]
    x = split.raw_test.iloc[row].to_numpy(dtype=float)
    braw = split.raw_train.median().to_numpy(dtype=float)
    xt = x.copy()
    for j in J:
        v = (1 - a) * x[j] + a * braw[j]
        codes = np.unique(split.raw_train.iloc[:, j].to_numpy())
        xt[j] = codes[np.argmin(np.abs(codes - v))]
    z = (x - split.mean_) / split.scale_
    zt = (xt - split.mean_) / split.scale_
    d = zt - z
    return TResult(zt, True, int(realized(d, eps).sum()), float(np.linalg.norm(d)))


def benign_order(split: Split) -> np.ndarray:
    """Binary features ordered by benign association in training data: P(x=1|benign) - P(x=1|threat)."""
    raw = split.raw_train.to_numpy(dtype=float)
    assoc = raw[split.y_train == 0].mean(axis=0) - raw[split.y_train == 1].mean(axis=0)
    return np.argsort(-assoc, kind="stable")


def add_binary(split: Split, row: int, n_add: int, rng: np.random.Generator, ranked: bool, eps: float,
               order: np.ndarray | None = None) -> TResult:
    """Set n_add absent binary features to 1.

    ranked=True : evasion-oriented capability shift, adding the absent permissions most associated
                  with the benign class (feature addition preserves app functionality; R2-M1).
    ranked=False: l0-matched control, adding randomly chosen absent binary features.
    """
    x = split.raw_test.iloc[row].to_numpy(dtype=float)
    absent = split.binary & (x == 0)
    if ranked:
        order = benign_order(split) if order is None else order
        pool = np.array([j for j in order if absent[j]], dtype=int)
    else:
        pool = np.flatnonzero(absent)
    if len(pool) < n_add:
        return TResult((x - split.mean_) / split.scale_, False, 0, 0.0, "too few absent features")
    S = pool[:n_add] if ranked else rng.choice(pool, size=n_add, replace=False)
    xt = x.copy()
    xt[S] = 1.0
    z = (x - split.mean_) / split.scale_
    zt = (xt - split.mean_) / split.scale_
    d = zt - z
    return TResult(zt, True, int(realized(d, eps).sum()), float(np.linalg.norm(d)))
