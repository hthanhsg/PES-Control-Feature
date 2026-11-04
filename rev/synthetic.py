"""Synthetic stand-ins with the same column conventions as the real sources (smoke tests only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .url_study import PhishVNSpec


def _naticus(rng, n=3000):
    d = 40
    w = rng.normal(0, 1.2, d)
    X = (rng.random((n, d)) < rng.uniform(0.05, 0.6, d)).astype(int)
    y = (X @ w + rng.normal(0, 1, n) > np.median(X @ w)).astype(int)
    cols = [f"android.permission.PERM_{i}" for i in range(d - 5)] + [f"com.vendor.permission.C{i}" for i in range(5)]
    df = pd.DataFrame(X, columns=cols)
    df["Result"] = y
    return df


def _uci(rng, n=3000):
    names = ["having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol", "double_slash_redirecting",
             "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State", "Domain_registeration_length", "Favicon", "port",
             "HTTPS_token", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH", "Submitting_to_email",
             "Abnormal_URL", "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe", "age_of_domain",
             "DNSRecord", "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page", "Statistical_report"]
    X = rng.choice([-1, 0, 1], size=(n, len(names)), p=[0.35, 0.2, 0.45])
    w = rng.normal(0, 1, len(names))
    y = np.where(X @ w + rng.normal(0, 1.5, n) > 0, -1, 1)
    df = pd.DataFrame(X, columns=names)
    df["result"] = y
    return df


def _url(rng, phish: bool):
    hosts = ["bank", "secure-login", "shop", "news", "mail", "pay", "account", "vn-gov", "portal", "cloud"]
    host = f"{rng.choice(['www.', '', 'm.', 'login.'])}{rng.choice(hosts)}{rng.integers(1, 500)}.{rng.choice(['com', 'vn', 'net', 'com.vn', 'xyz'])}"
    path = "" if (not phish and rng.random() < 0.6) else "/" + "/".join(rng.choice(["a", "verify", "id", "x1", "update"], rng.integers(1, 4)))
    q = f"?id={rng.integers(1, 99999)}" if rng.random() < (0.5 if phish else 0.1) else ""
    return f"https://{host}{path}{q}"


def _phiusiil(rng, n=4000):
    y_legit = (rng.random(n) < 0.57).astype(int)
    urls = [_url(rng, phish=not bool(l)) for l in y_legit]
    df = pd.DataFrame({"FILENAME": [f"f{i}.txt" for i in range(n)], "URL": urls,
                       "URLLength": [len(u) for u in urls], "Domain": [u.split('/')[2] for u in urls],
                       "DomainLength": [len(u.split('/')[2]) for u in urls], "IsDomainIP": 0, "TLD": "com",
                       "URLSimilarityIndex": np.where(y_legit == 1, 100, rng.uniform(20, 100, n)),
                       "CharContinuationRate": rng.random(n), "NoOfSubDomain": rng.integers(0, 3, n),
                       "HasTitle": rng.integers(0, 2, n), "HasDescription": rng.integers(0, 2, n),
                       "NoOfiFrame": rng.poisson(1, n), "HasExternalFormSubmit": rng.integers(0, 2, n),
                       "HasSocialNet": y_legit * rng.integers(0, 2, n), "HasPasswordField": rng.integers(0, 2, n),
                       "Bank": rng.integers(0, 2, n), "Pay": rng.integers(0, 2, n), "NoOfImage": rng.poisson(10, n),
                       "NoOfJS": rng.poisson(5, n), "NoOfSelfRef": rng.poisson(20, n) * (1 + y_legit),
                       "LineOfCode": rng.poisson(500, n) * (1 + y_legit), "label": y_legit})
    for i in range(10):
        df[f"Extra{i}"] = rng.normal(0, 1, n) + 0.3 * y_legit * (i % 2)
    return df


def write_all(data_dir: Path) -> PhishVNSpec:
    rng = np.random.default_rng(0)
    data_dir.mkdir(parents=True, exist_ok=True)
    _naticus(rng).to_csv(data_dir / "naticusdroid.csv", index=False)
    _uci(rng).to_csv(data_dir / "uci_phishing.csv", index=False)
    _phiusiil(rng).to_csv(data_dir / "phiusiil.csv", index=False)
    n = 3000
    lab = (rng.random(n) < 0.15).astype(int)
    vn = pd.DataFrame({"url": [_url(rng, bool(l)).replace(".com", ".vn") + f"/{i}" for i, l in enumerate(lab)],
                       "label": np.where(lab == 1, "phishing", "benign"),
                       "confidence_tier": rng.choice(["gold", "silver"], n),
                       "first_seen": pd.date_range("2024-01-01", periods=n, freq="h")})
    vn.to_csv(data_dir / "phishvn.csv", index=False)
    return PhishVNSpec(path=str(data_dir / "phishvn.csv"))
