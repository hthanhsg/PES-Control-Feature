"""External URL study (Sec. 5.2 / 6.3) with the additions requested in review.

  * cross-source PhiUSIIL -> PhishVN with full metrics, confusion matrices, calibration (R2-M3 ii)
  * in-target PhishVN reference: domain-grouped and temporal splits (R2-M3 i)
  * per-class distributions of lexical features in both sources (R2-M3 iii)
  * gold / silver tiers reported separately (R2-M3 iv)
  * fragment edits: explicit generator, standardized fragment length, flips by direction (R2-M4)
  * attributions for the predicted class as well as the recorded true class (R2-M6)
  * unpaired attribution-distribution distances with n and bootstrap intervals (R3 minor, Fig. 6)

PhishVN columns are configurable (see PhishVNSpec). CONFIRM them against the released CSV.
"""
from __future__ import annotations

import logging
import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .common import Config
from .data import load_raw
from .explainers import ablation
from .metrics import k_of, topk_instability
from .models import make_model, proba
from .transforms import redistribute

LOG = logging.getLogger("rev")
FEATURES = ["url_len", "host_len", "path_len", "query_len", "fragment_len", "n_dot", "n_slash", "n_digit",
            "n_hyphen", "n_underscore", "n_at", "n_percent", "n_qmark", "n_equal", "n_subdomain", "is_ip", "entropy"]
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$")


@dataclass
class PhishVNSpec:
    path: str = "data/phishvn.csv"
    url_col: str = "url"
    label_col: str = "label"
    threat_values: tuple = ("phishing", "1", 1)        # CONFIRM
    tier_col: str | None = "confidence_tier"            # CONFIRM (gold / silver / ...)
    keep_tiers: tuple = ("gold", "silver")
    time_col: str | None = "first_seen"                 # CONFIRM; None disables the temporal split
    channel_col: str | None = "channel"                 # PhishVN mixes channels (url, sms, ...)
    keep_channels: tuple | None = None                  # e.g. ("url",); None keeps all, but the mix is logged


def _tld():
    import tldextract
    return tldextract.TLDExtract(suffix_list_urls=())   # offline snapshot: reproducible PSL


_EXTRACT = None


def registrable(url: str) -> str:
    global _EXTRACT
    _EXTRACT = _EXTRACT or _tld()
    e = _EXTRACT(url if SCHEME.match(url) else "http://" + url)
    rd = getattr(e, "top_domain_under_public_suffix", None) or e.registered_domain
    return (rd or e.domain or url).lower()


def normalize(url: str) -> str:
    u = SCHEME.sub("", str(url).strip().lower())
    u = u[4:] if u.startswith("www.") else u
    return u.rstrip("/")


def entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def features(url: str) -> list[float]:
    global _EXTRACT
    _EXTRACT = _EXTRACT or _tld()
    rest = SCHEME.sub("", str(url).strip())
    parts = urlsplit("//" + rest)
    host = parts.hostname or ""
    sub = _EXTRACT("http://" + rest).subdomain
    return [len(rest), len(host), len(parts.path), len(parts.query), len(parts.fragment),
            rest.count("."), rest.count("/"), sum(ch.isdigit() for ch in rest), rest.count("-"), rest.count("_"),
            rest.count("@"), rest.count("%"), rest.count("?"), rest.count("="),
            len(sub.split(".")) if sub else 0, float(bool(IPV4.match(host)) or host.startswith("[")), entropy(rest)]


def featurize(urls) -> np.ndarray:
    return np.array([features(u) for u in urls], dtype=float)


def edit_fragment(url: str, rng: np.random.Generator, token_len: int = 8, prefix: str = "section-") -> str:
    """Request-target-preserving edit: append (or replace) a URL fragment.  CONFIRM generator."""
    tok = "".join(rng.choice(list(string.ascii_lowercase + string.digits), size=token_len))
    base = str(url).split("#", 1)[0]
    return f"{base}#{prefix}{tok}"


def same_request_target(u1: str, u2: str) -> bool:
    a, b = urlsplit(u1 if SCHEME.match(u1) else "http://" + u1), urlsplit(u2 if SCHEME.match(u2) else "http://" + u2)
    return (a.scheme, a.netloc, a.path, a.query) == (b.scheme, b.netloc, b.path, b.query)


def ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics(y, p, thr: float = 0.5) -> dict:
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(n=len(y), prevalence=float(np.mean(y)), ba=balanced_accuracy_score(y, yh), f1=f1_score(y, yh, zero_division=0),
                auc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                pr_auc=average_precision_score(y, p) if len(set(y)) > 1 else np.nan,
                brier=brier_score_loss(y, p), ece=ece(np.asarray(y), p), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1), pred_pos_rate=float(yh.mean()),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                benign_p_gt_09=float(np.mean(p[np.asarray(y) == 0] > 0.9)) if (np.asarray(y) == 0).any() else np.nan)


def to_host(url: str) -> str:
    """Reduce a URL to its host name (format-matched comparison with domain-only PhishVN records)."""
    rest = SCHEME.sub("", str(url).strip())
    return (urlsplit("//" + rest).hostname or rest.split("/")[0]).lower()


def load_sources(cfg: Config, spec: PhishVNSpec, host_only: bool = False):
    X, y = load_raw("phiusiil", cfg.data_dir)
    ucol = next(c for c in X.columns if c.lower() == "url")
    src = pd.DataFrame({"url": X[ucol].astype(str), "y": y.to_numpy()})
    if host_only:
        src["url"] = src["url"].map(to_host)
        LOG.info("host-only: PhiUSIIL URLs reduced to host names (format-matched to PhishVN)")
    raw = pd.read_csv(spec.path, low_memory=False)
    n_raw = len(raw)
    raw = raw[raw[spec.url_col].notna() & (raw[spec.url_col].astype(str).str.strip() != "")]
    LOG.info("PhishVN: dropped %d rows without a URL", n_raw - len(raw))
    if spec.channel_col and spec.channel_col in raw:
        LOG.info("PhishVN channels: %s", raw[spec.channel_col].value_counts().to_dict())
        if spec.keep_channels:
            raw = raw[raw[spec.channel_col].astype(str).isin(spec.keep_channels)]
    has_path = raw[spec.url_col].astype(str).map(lambda u: "/" in SCHEME.sub("", u)).mean()
    LOG.info("PhishVN: share of URLs with a path component: %.3f", has_path)
    tgt = raw
    tgt = pd.DataFrame({"url": tgt[spec.url_col].astype(str),
                        "y": tgt[spec.label_col].astype(str).isin([str(v) for v in spec.threat_values]).astype(int),
                        "tier": tgt[spec.tier_col].astype(str).str.lower() if spec.tier_col and spec.tier_col in tgt else "all",
                        "time": pd.to_datetime(tgt[spec.time_col], errors="coerce") if spec.time_col and spec.time_col in tgt else pd.NaT})
    tgt = tgt[tgt["tier"].isin(spec.keep_tiers) | (tgt["tier"] == "all")]
    for df in (src, tgt):
        df["norm"] = df["url"].map(normalize)
        df["rd"] = df["url"].map(registrable)
    before = len(tgt)
    tgt = tgt[~tgt["norm"].isin(set(src["norm"])) & ~tgt["rd"].isin(set(src["rd"]))].reset_index(drop=True)
    LOG.info("PhishVN: %d -> %d after overlap removal (paper: 17,938)", before, len(tgt))
    return src, tgt


def run(cfg: Config, spec: PhishVNSpec, out: Path, models=("lr", "rf"), frag_per_seed: int = 200,
        attr_per_class: tuple = (24, 200), host_only: bool = False) -> None:
    src_all, tgt = load_sources(cfg, spec, host_only=host_only)
    Ft = featurize(tgt["url"])
    perf, intarget, frag_rows, dist_rows, attr_rows = [], [], [], [], []
    seeds = cfg.seeds[:2] if cfg.quick else cfg.seeds
    for seed in seeds:
        src, _ = train_test_split(src_all, train_size=min(12_000, len(src_all) - 2), stratify=src_all["y"],
                                  random_state=seed) if len(src_all) > 12_000 else (src_all, None)
        src = src.reset_index(drop=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        tr, te = next(gss.split(src, src["y"], groups=src["rd"]))
        Fs = featurize(src["url"])
        mu, sd = Fs[tr].mean(0), Fs[tr].std(0)
        sd[sd == 0] = 1
        Z = lambda F: (F - mu) / sd
        b = np.median(Z(Fs[tr]), axis=0)
        # distributions (R2-M3 iii) -- once per seed is enough; keep first seed
        if seed == seeds[0]:
            for name, F, yy in (("PhiUSIIL", Fs, src["y"].to_numpy()), ("PhishVN", Ft, tgt["y"].to_numpy())):
                for c in (0, 1):
                    for j, f in enumerate(FEATURES):
                        v = F[yy == c, j]
                        dist_rows.append(dict(source=name, cls=c, feature=f, mean=v.mean(), median=np.median(v),
                                              frac_nonzero=float((v > 0).mean())))
            for c in (0, 1):
                for j, f in enumerate(FEATURES):
                    dist_rows.append(dict(source="KS PhiUSIIL vs PhishVN", cls=c, feature=f,
                                          mean=ks_2samp(Fs[src["y"].to_numpy() == c, j], Ft[tgt["y"].to_numpy() == c, j]).statistic))
        for kind in models:
            m = make_model(kind, seed, url=True).fit(Z(Fs[tr]), src["y"].to_numpy()[tr])
            ps, pt = proba(m, Z(Fs[te])), proba(m, Z(Ft))
            perf.append(dict(seed=seed, model=kind, population="PhiUSIIL holdout", **metrics(src["y"].to_numpy()[te], ps)))
            perf.append(dict(seed=seed, model=kind, population="PhishVN all", **metrics(tgt["y"].to_numpy(), pt)))
            for tier in sorted(set(tgt["tier"]) - {"all"}):
                mk = (tgt["tier"] == tier).to_numpy()
                perf.append(dict(seed=seed, model=kind, population=f"PhishVN {tier}", **metrics(tgt["y"].to_numpy()[mk], pt[mk])))
            # ---- fragment edits (R2-M4)
            rng = np.random.default_rng([seed, 99])
            te_idx = np.array(te)
            pick = te_idx if len(te_idx) <= frag_per_seed else rng.choice(te_idx, size=frag_per_seed, replace=False)
            urls = src["url"].to_numpy()[pick]
            edited = [edit_fragment(u, rng) for u in urls]
            ok = np.array([same_request_target(u, e) for u, e in zip(urls, edited)])
            Z0, Z1 = Z(featurize(urls)), Z(featurize(edited))
            frag_j = FEATURES.index("fragment_len")
            Zc = np.vstack([redistribute(Z0[i], Z1[i] - Z0[i], rng, cfg.move_eps).z for i in range(len(pick))])
            p0, p1, pc = proba(m, Z0), proba(m, Z1), proba(m, Zc)
            cls = (p0 >= 0.5).astype(int)
            yy = src["y"].to_numpy()[pick]
            kk = k_of(len(FEATURES), cfg.k_frac)
            for target_name, tcls in (("predicted", cls), ("true", yy)):
                A0, A1, Ac = (ablation(m, Zx, b, tcls, "prob") for Zx in (Z0, Z1, Zc))
                for i in range(len(pick)):
                    frag_rows.append(dict(seed=seed, model=kind, target=target_name, url_id=int(pick[i]), y=int(yy[i]),
                                          request_target_ok=int(ok[i]), p0=p0[i], p_frag=p1[i], p_ctrl=pc[i],
                                          flip=int((p1[i] >= 0.5) != (p0[i] >= 0.5)),
                                          direction=("threat->benign" if p0[i] >= 0.5 > p1[i] else
                                                     "benign->threat" if p1[i] >= 0.5 > p0[i] else "none"),
                                          D_frag=topk_instability(A0[i], A1[i], kk, cfg.tie_abs, cfg.tie_rel),
                                          D_ctrl=topk_instability(A0[i], Ac[i], kk, cfg.tie_abs, cfg.tie_rel),
                                          z_fragment_len=Z1[i, frag_j],
                                          train_frac_fragment=float((Fs[tr][:, frag_j] > 0).mean())))
            # ---- unpaired attribution distributions (Fig. 6) for two sample sizes
            for n_per in attr_per_class:
                for c in (0, 1):
                    s_idx = te_idx[src["y"].to_numpy()[te_idx] == c]
                    t_idx = np.flatnonzero(tgt["y"].to_numpy() == c)
                    if len(s_idx) < 2 * n_per or len(t_idx) < n_per:
                        continue
                    s_pick = rng.choice(s_idx, size=2 * n_per, replace=False)
                    t_pick = rng.choice(t_idx, size=n_per, replace=False)
                    def mean_norm(Zx, tc):
                        A = np.abs(ablation(m, Zx, b, tc, "prob"))
                        A = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
                        return A
                    As1 = mean_norm(Z(Fs[s_pick[:n_per]]), np.full(n_per, c))
                    As2 = mean_norm(Z(Fs[s_pick[n_per:]]), np.full(n_per, c))
                    At = mean_norm(Z(Ft[t_pick]), np.full(n_per, c))
                    boot = []
                    for _ in range(500):
                        i1, i2 = rng.integers(0, n_per, n_per), rng.integers(0, n_per, n_per)
                        boot.append(np.abs(As1[i1].mean(0) - At[i2].mean(0)).sum())
                    attr_rows.append(dict(seed=seed, model=kind, cls=c, n_per=n_per,
                                          within=np.abs(As1.mean(0) - As2.mean(0)).sum(),
                                          across=np.abs(As1.mean(0) - At.mean(0)).sum(),
                                          across_lo=np.quantile(boot, 0.025), across_hi=np.quantile(boot, 0.975)))
        # ---- in-target reference (R2-M3 i)
        gss_t = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
        ttr, tte = next(gss_t.split(tgt, tgt["y"], groups=tgt["rd"]))
        splits = [("PhishVN grouped", ttr, tte)]
        if tgt["time"].notna().mean() > 0.9:
            order = np.argsort(tgt["time"].to_numpy())
            cut = int(0.7 * len(order))
            a, bidx = order[:cut], order[cut:]
            seen = set(tgt["rd"].to_numpy()[a])
            bidx = np.array([i for i in bidx if tgt["rd"].iat[i] not in seen])
            splits.append(("PhishVN temporal", a, bidx))
        for sname, a, bidx in splits:
            mu2, sd2 = Ft[a].mean(0), Ft[a].std(0)
            sd2[sd2 == 0] = 1
            for kind in models:
                m2 = make_model(kind, seed, url=True).fit((Ft[a] - mu2) / sd2, tgt["y"].to_numpy()[a])
                intarget.append(dict(seed=seed, model=kind, population=sname,
                                     **metrics(tgt["y"].to_numpy()[bidx], proba(m2, (Ft[bidx] - mu2) / sd2))))
    d = out / "url"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(perf).to_csv(d / "performance.csv", index=False)
    pd.DataFrame(intarget).to_csv(d / "in_target.csv", index=False)
    pd.DataFrame(frag_rows).to_csv(d / "fragment.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(d / "feature_distributions.csv", index=False)
    pd.DataFrame(attr_rows).to_csv(d / "attribution_distance.csv", index=False)
    LOG.info("URL study done: %d PhishVN URLs after overlap removal", len(tgt))
