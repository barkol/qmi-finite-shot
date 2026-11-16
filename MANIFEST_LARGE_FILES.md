# Large files not stored in the repository

These files are not stored because of their size. Trained networks and intermediate files are available from the
authors on request, and the scripts regenerate them from fixed seeds. Paths are relative to the working directory of
the pipeline (`QMI_ROOT`, by default the repository root). Sizes are approximate.

## A. Trained networks

| path | size | content |
|---|---|---|
| `runs_eq/` | 1.7 GB | 78 networks of the base set (`qmi/eq/plan_eq.py`), with `history.json` and evaluation outputs |
| `runs_ahm_eq/` | 0.50 GB (weights 0.46 GB) | 54 networks of the second set (Ahmed et al.) |

## B. Intermediate results

| path | size | content |
|---|---|---|
| `results_eq/scores.json`, `results_ahm_eq/scores.json` | 20 MB, 16 MB | per-state network outputs (their summaries `scores_summary.json` are stored) |
| `analyses/lr_observer/cache/`, `analyses/lr_observer/parts/` | small | prototype libraries and partial predictions of the likelihood-ratio observer |

## C. Generated data sets (regenerated from the seeds)

| path | size | content |
|---|---|---|
| `data_eq/` | 7.6 GB | exact images and finite records of the base set (`qmi.gen --family eqn`, `shots_data`, `sino_data`) |
| `data_eq_ahm/` | 6.8 GB | the same for the second set |
