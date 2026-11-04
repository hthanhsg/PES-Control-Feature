# Datasets

No raw dataset is redistributed in this repository. Place the four CSV files
below in this directory (or let `run_all.py` download the UCI sets).

| File | Source | How to obtain |
|---|---|---|
| `naticusdroid.csv` | UCI dataset 722 (NATICUSdroid) | auto-downloaded via `ucimlrepo`, or export manually |
| `uci_phishing.csv` | UCI dataset 327 (Phishing Websites) | auto-downloaded via `ucimlrepo` |
| `phiusiil.csv` | UCI dataset 967 (PhiUSIIL) | auto-downloaded via `ucimlrepo` |
| `phishvn.csv` | PhishVN release cited in the paper | download from the data record cited as [vu2026]; not redistributed here |

The official runs were made from fixed CSV snapshots. Their SHA-256 checksums
are recorded in the provenance records of the deposit:

```
naticusdroid.csv  9af2bbc9499f2b3520a5e70d59b809374aa7350ad92c2814aa3b68292a93b457
uci_phishing.csv  a13fc4835e437bb39e5ddce9ae022af18f67f504332e3383b224c599dd721c40
phiusiil.csv      cb4e29fc9a16e20c6f9b3a157ac68e6605be2515b5b641211642d80fdde2169d
phishvn.csv       8e1c6325a82177014ac892a9aaf8c4d3cbbc986b47fd1ecac8899c865fb6eade
```

If a freshly downloaded file does not match these checksums, results may
differ slightly from the published tables; the paper's numbers correspond to
the recorded snapshots.
