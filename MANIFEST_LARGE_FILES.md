# Large artifacts not stored in git

These files are not stored in git (size). Trained networks and intermediate files are not archived; they are available
from the authors on request, and the scripts regenerate them from fixed seeds. Paths are relative to the working
directory of the pipeline (the repository root, where `runs_eq/`, `data_eq/` etc. are created); `analyses/<name>/`
denotes the directory of the respective analysis in this repository. Sizes are approximate.

## A. Trained networks (needed to rerun any evaluation without retraining)

| path | size | content |
|---|---|---|
| `runs_eq/` | 1.7 GB (weights `best.pt` 0.69 GB) | 81 networks of the base set eq (`qmi/eq/plan_eq.py`), with `history.json`, evaluation outputs and `konno/` (0.5 MB, also copied to `paper/results/konno/`) |
| `runs_eq/clf_W_shotsK6_s{0,1,2}`, `runs_eq/clf_W_gkpxK6_s{0,1,2}` | 52 MB (part of `runs_eq/`) | K=6 networks used for the Konno and Kawasaki records |
| `runs_ahm_eq/` | 0.50 GB (weights 0.46 GB) | 54 networks of the Ahmed set (eq noise, window 12) |
| `analyses/kawasaki/runs_ahmK6/` | 102 MB | 3 Ahmed networks with K=6 (Kawasaki analysis; not used in the text) |
| `analyses/maxlik/runs/`, `runs_ahm/`, `runs_D20/` | 3 x 102 MB | 9 classifiers on MaxLik reconstructions (D=40 base, Ahmed; D=20 control) |
| `analyses/arch2/runs/` | 385 MB | 9 ResNet-18 networks (Q shots, W shots, W sino; seeds 0-2) |

## B. Intermediate results of the analyses (regenerable, but slow)

| path | size | content |
|---|---|---|
| `analyses/maxlik/ev_base/`, `ev_ahm/` | 0.1 GB each | events of the paired test records |
| `analyses/maxlik/ml_base/`, `ml_ahm/`, `ml_base_D20/` | 0.14 GB each | MaxLik states of the test records |
| `analyses/maxlik/trev/`, `trev_ahm/` | 0.68 GB each | events and MaxLik states of the training records |
| `analyses/maxlik/cls_*/` | 1 MB each | correctness matrices (input of `analyze.py`) |
| `analyses/maxlik/wbasis_D30.npy` etc. | 0.11 GB | Wigner basis cache |
| `analyses/arch2/det/`, `analyses/arch2/pary/` | 13 MB, 0.2 MB | test caches and paired correctness matrices |
| `analyses/chernoff/out/M24_*.npz` | 2.3 MB | per-state Chernoff exponents (input of `podsumuj.py`) |
| `analyses/kawasaki/cache_ahmK6/` | 2.2 GB | K=6 records of the Ahmed set (training cache) |
| `analyses/kawasaki/out/*/`, `out_konno/*.npz` | small | per-file npz of the Kawasaki and Konno record tests |
| `analyses/kawasaki_epr/dane/` | 0.51 GB | quadratures extracted from the EPR traces (`krok12_kwadratury.py`) |
| `analyses/shap/out/` | 1.6 GB | attribution maps per cell (`attr_*.npz`, `maps.npz`) |
| `paper/figures/data/fig_states.npz` | 8.9 MB | cache of `fig_states.py` (above the 5 MB limit for git) |
| `results_eq/scores.json`, `results_ahm_eq/scores.json` | 20 MB, 16 MB | per-state network outputs (summaries `scores_summary.json` are in git) |

## C. Generated datasets (regenerable from the seeds)

| path | size | content |
|---|---|---|
| `data_eq/` | 7.6 GB | exact images and finite records of the base set eq (`qmi.gen --family eqn`, `shots_data`, `sino_data`) |
| `data_eq_ahm/` | 6.8 GB | the same for the Ahmed set |

## D. Experimental data (third party, CC0, not redistributed; cite the Dryad DOIs)

| path | size | source |
|---|---|---|
| `data/dryad_gkp/quad_*deg.npy` | 1 MB | Konno et al. 2024, https://doi.org/10.5061/dryad.t76hdr86j |
| `dane_eksp/kawasaki2024/raw/` | 2.6 MB | Kawasaki et al. 2024, https://doi.org/10.5061/dryad.9p8cz8wqn |
| `dane_eksp/kawasaki2024_epr/raw/` | 6.1 GB | Kawasaki et al. 2025, https://doi.org/10.5061/dryad.dbrv15fbq |
