# Reporting checklist for explanation-stability claims about security detectors

This is the checklist released with the paper *"Prediction and Explanation
Sensitivity under Controlled Feature Shifts in Malware and Phishing
Detection"* (Table 4 of the conference version). Each item names what a study
should report; the last column quantifies, on the paper's benchmarks, how much
the reported contrast can change when the item is omitted.

An explanation-stability claim compares attributions of a clean record with
attributions of a transformed record. The claim is interpretable only when the
reader can separate the effect of interest from explainer noise, displacement
size, operation and direction, probability response, and ranking ties.

| # | Report | Why | Largest effect of omission observed in the paper |
|---|--------|-----|--------------------------------------------------|
| 1 | **Explainer, reference point, score scale, sampling budget and seeds.** | Attributions are functions of the removal distribution and reference; TreeSHAP and ablation differ systematically. | Up to 0.11 difference in the contrast (TreeSHAP vs. ablation). |
| 2 | **Identity floor of sampled explainers at the budget actually used** — the instability between two runs on the *same* input (LIME, KernelSHAP). | Any reported instability below the identity floor is indistinguishable from explainer noise. | Floors up to 0.57 at 128 samples; 0.01–0.13 at library defaults. |
| 3 | **Chance level, attainable values and tie rule of the similarity measure.** | Top-k Jaccard on d features takes few discrete values; ties at the cutoff can dominate. | Chance level E[D] = 0.898 for d = 24, k = 4; ties at the cutoff in about half of the NATICUSdroid explanations and in all saturated GB/PhiUSIIL explanations. |
| 4 | **A displacement-matched control** (same number of changed coordinates and the same Euclidean displacement, placed at random). | Raw instability confounds the transformation with its size. | Matched contrasts range from −0.29 to +0.51 where raw numbers would suggest a single story. |
| 5 | **An operation- and direction-matched control** (the same operation, e.g. shrinkage towards the reference, on random coordinates). | Redistribution moves coordinates *away* from the reference; for linear models this alone reverses the sign. | Changes up to 0.31; the negative logistic-regression contrast (−0.285) disappears (+0.006), identifying it as a control artefact. |
| 6 | **The probability response alongside every contrast**, plus either a probability-matched control or a record-level model estimating the contrast at zero probability response. | Explanations may change *because* the score changed; this share should not be sold as explanation instability. | Changes up to 0.44; for tree ensembles 20–90% of the unadjusted contrast accompanies larger prediction changes. |
| 7 | **Results per detector family, never pooled.** | The sign and size of the contrast are model-dependent. | The pooled value (+0.16) describes no cell; cells range from −0.29 to +0.51. |
| 8 | **The feature-domain and problem-space status of every shift** (numerically valid? valid feature encoding? functionality-preserving in the problem space?). | Off-manifold vectors and infeasible edits do not support attack or deployment claims. | A feature-domain-valid variant of the mask turns the LR contrast from ≈0 to +0.21; none of the paper's shifts is problem-space validated. |
| 9 | **An in-target reference for every cross-source result** (the same representation trained and tested within the target). | Without it, transfer failure cannot be separated from an uninformative representation. | Cross-source balanced accuracy 0.53–0.57 vs. 0.76–0.78 in-target: the representation carries signal, the transfer fails. |

## How this repository supports each item

| # | Where |
|---|-------|
| 1 | `rev/explainers.py` (ablation, SHAP variants, references, scales); seeds in `rev/common.py` (`Config`). |
| 2 | `rev/source_study.py` stochastic-floor stage → `tab_stochastic.tex`. |
| 3 | `rev/metrics.py` (`topk_instability`, tie weights, chance level) → `tab_ties.tex`, `tab_ksens.tex`. |
| 4 | `rev/transforms.py::redistribute` → "Redistr." column of `tab_ladder.tex`. |
| 5 | `rev/transforms.py::op_matched`, `mi_matched`; `run_full_matched.py` → "Operation", "MI rank" columns; `tab_fullmatch.tex`. |
| 6 | `rev/transforms.py::dp_matched`; record-level mixed model in `rev/tables.py::mixed_model` → `tab_mixed.tex`, γ = 0 column. |
| 7 | All tables are stratified by dataset × model; no pooled rows are produced. |
| 8 | `rev/transforms.py::mask_discrete` (feature-domain-valid mask), `add_perm` (capability-constrained shift); problem-space status stated in the paper. |
| 9 | `rev/url_study.py` in-target PhishVN models → `tab_external.tex`. |
