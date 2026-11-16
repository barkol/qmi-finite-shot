# qmi-finite-shot

Code, scripts and result files for the paper

> F. Grzywa, S. Dolata and K. Bartkiewicz, *Homodyne versus heterodyne detection near Gaussian states: Fraction of the
> quantum Fisher information recovered* (submitted).

Authors: see `AUTHORS` and `CITATION.cff`. Contact: bark@amu.edu.pl. License: MIT (see `LICENSE`).

The code depends on the qsdl library (https://github.com/Askartan/qsdl, commit a11056285ad9), which is not included.
The pinned commit has no license file; the current head of qsdl is under the MIT license.

## Repository layout

| path | content |
|---|---|
| `analyses/fisher/` | numerical checks of the Fisher-information results of Sec. II and Appendix A (direct integrals of the outcome densities, independent of the formulas being checked); see `analyses/fisher/README.md` |
| `analyses/chernoff/` | Chernoff exponents of heterodyne and homodyne detection ($K=36$ phases) and quantum Chernoff exponents for the pairs of states of the test (Sec. III A); pair matrices `out/M24_*.npz`, summary `out/summary_M24.json` |
| `analyses/lr_observer/` | likelihood-ratio observer on the records of the base set (Sec. III B); result file `lr_results.json`, tables `make_tables.py` |
| `qmi/` | simulation and network pipeline of Sec. III: state families (`eqn_states.py` base set, `ahmed_states.py` second set), finite records of $N$ events and filtered back-projection (`shots.py`, `shots_data.py`, `sino_data.py`), training (`clf.py`), evaluation (`eval_shots.py`, `eval_scores.py`, `eval_drift.py`), baselines, energy control, quantum limit (`limit_sdp.py`, `limit_sigma.py`) |
| `qmi/eq/` | plan of stages and networks (`plan_eq.py`), resumable queue (`queue_eq.py`), final step (`finish_eq.py`), homodyne networks with and without reconstruction on the same events (`pairs_sw.py`), FBP window check (`fbp_window.py`) |
| `qmi/x8/` | evaluation-stage modules: data guard (`guard.py`), paired evaluation (`pairs.py`), derived files (`derived.py`) |
| `sm/plan.py` | stage and network plan (seeds, sizes, arguments) from which `qmi/eq/plan_eq.py` derives the plan of the run (78 networks of the base set, 54 of the second set) |
| `paper/results/`, `paper/results_ahm/` | result files of the base set and of the second set; `numbers*.tex` (LaTeX macros with the numbers of the paper) and the intervals `ci*.json` |
| `paper/scripts/` | number generators `make_numbers.py`, `numbers_fisher.py`, `holm_rule.py`; comparison tool `json_close.py` |
| `paper/figures/` | figure scripts: one directory per figure with a data layer `data_*.py` and a drawing layer `plot_*.py`; colour roles `palette.py`, matplotlib style `figstyle.py`, data caches in `paper/figures/data/` |
| `reproduce_numbers.sh` | regenerates the number files from the result files and compares them with the committed versions |
| `MANIFEST_LARGE_FILES.md` | large files that are not stored (networks, data sets) |

## Reproduction

### Numbers (under a minute, numpy and scipy only)

    ./reproduce_numbers.sh            # PY=<python with numpy and scipy>, default python3

The script runs

    cd paper
    python scripts/make_numbers.py                                           # results/numbers.tex, results/ci.json
    RES=results_ahm PFX=ahm_ OUTDIR=results python scripts/make_numbers.py   # results/ci_ahm.json (second set)
    python scripts/numbers_fisher.py                                         # results/numbers_fisher.tex
    cd ..
    python analyses/chernoff/numbers_chernoff.py                             # paper/results/numbers_chernoff.tex
    python paper/scripts/holm_rule.py paper/results                          # paper/results/numbers_holm.tex

and compares every `numbers*.tex` (byte by byte) and `ci*.json` (relative tolerance 1e-9) with the committed version;
it exits with code 1 if any file differs. The numbers of the likelihood-ratio observer (Sec. III B, Fig. 8) are read
from `analyses/lr_observer/lr_results.json` (`python analyses/lr_observer/make_tables.py` prints them as tables).

### Figures

Each figure has a data layer (`data_*.py`, computes the cache in `paper/figures/data/` if it is missing) and a drawing
layer (`plot_*.py`, reads only the cache). Run the drawing script; it builds the cache if needed.

| figure | directory in `paper/figures/` | drawing | input of the data layer |
|---|---|---|---|
| 1 | `fig_gauss_overview/` | matplotlib | closed forms |
| 2 | `fig_modes/` | matplotlib | closed forms |
| 3 | `fig_mode_efficiency/` | GLE | closed forms |
| 4 | `fig_squeezed_bounds/` | GLE | `analyses/fisher/symlib.py` |
| 5 | `fig_tmsv/` | GLE | closed forms |
| 6 | `fig_disp_chernoff/` | GLE | closed forms |
| 7 | `fig_dephasing_sectors/` (`plot_fig_dephasing_prediction.py`) | GLE | `analyses/chernoff/out/M24_*.npz`, `analyses/fisher/out/c18.json`, `c19.json` (rebuilding the cache needs qsdl) |
| 8 | `fig_observer/` | GLE | `analyses/lr_observer/lr_results.json` |

The GLE figures need GLE 4.3 (glx.sourceforge.io) with LaTeX; the matplotlib figures use `figstyle.py` (LaTeX text).
All figures take their colours from `palette.py` (TURBO colormap with fixed roles); every series also differs by line
style or marker.

### Checks of Sec. II

Run the scripts of `analyses/fisher/` from that directory; each writes `out/<name>.json` and the long ones resume from
their partial results. `paper/scripts/numbers_fisher.py` turns `out/*.json` into `numbers_fisher.tex`.

### Chernoff exponents and likelihood-ratio observer

    QMI_XMAX=12 PYTHONPATH=$PWD:<qsdl> python analyses/chernoff/chernoff.py --M 24     # out/M24_*.npz (seed 2210)
    python analyses/chernoff/summarize.py M24                                       # out/summary_M24.json
    QMI_XMAX=12 QMI_ROOT=<dir with data_eq/> PYTHONPATH=$PWD:<qsdl> python analyses/lr_observer/lr_observer.py all

The observer reads the test states of the base set (`data_eq/`, regenerated by the pipeline below) and writes
resumable part files before `lr_results.json`.

### From scratch (days on one GPU)

The data sets (about 15 GB) are not stored; they are regenerated deterministically from the seeds of the plan:

    QMI_XMAX=12 python -m qmi.eq.plan_eq                        # base set: stages and networks
    QEQ_SET=ahm QMI_XMAX=12 python -m qmi.eq.plan_eq            # second set
    QMI_ROOT=$PWD QMI_XMAX=12 python -m qmi.eq.queue_eq         # data, training, evaluations, baselines (resumable)
    QMI_ROOT=$PWD QMI_XMAX=12 python qmi/eq/finish_eq.py        # derived files -> results_eq/

`queue_eq.py` is resumable (marker files in `data_eq/_markers/`) and stops on the first error. During production some
networks were trained on cloud GPU instances with a job wrapper that is not part of this repository (`QEQ_TRAIN_RX`
selects the networks trained locally); without it every network is trained locally. The quantum limit
(`limit_loss_dim48.json`) comes from `qmi/limit_sdp.py` with the MOSEK solver. GPU training is not claimed to
reproduce the networks bit by bit on other hardware.

## Environment

    git clone https://github.com/Askartan/qsdl && git -C qsdl checkout a11056285ad9d684eb09b982b9c3d88841abe77b
    pip install -r requirements.txt
    export PYTHONPATH=$PWD:$PWD/qsdl

qsdl at this commit cannot be installed with pip and asks for qutip>=5.3, whereas the results were produced with
qutip 5.2.3; hence the clone and `PYTHONPATH`. For bitwise reproduction of generated images set `OMP_NUM_THREADS=1`.

**Phase-space window.** The pinned qsdl commit fixes `XMAX = 5.0` in `qsdl/config.py`; all results use `QMI_XMAX=12`.
`qmi/__init__.py` sets `qsdl.config.XMAX` from this variable before any qsdl module that copies `XMAX` at import time
is loaded (and raises an error if such a module was imported first). Run the code as `QMI_XMAX=12 python -m qmi.<module>`;
scripts in `analyses/` import `qmi` before `qsdl`. `qmi.x8.guard` checks the window stored in every generated file.

| variable | meaning | default |
|---|---|---|
| `QMI_XMAX` | half-width of the phase-space window | unset = 5.0 of qsdl; use `12` |
| `QMI_ROOT` | working directory with the large inputs (`data_eq/`, `data_eq_ahm/`, `runs_eq/`, `runs_ahm_eq/`, `results_eq/`) | repository root |
| `QMI_PAPER` | directory written by the number generators | `paper/` of the repository |
| `QMI_PY` | interpreter used by the queue | the running interpreter |

Python 3.12 with the versions in `requirements.txt`; `reproduce_numbers.sh` needs only numpy and scipy.

## Funding

This work was supported by the Excellence Initiative – Research University (IDUB) program of Adam Mickiewicz
University (Advanced Best Student Grant, Edition 5). K.B. acknowledges support from the EuroHPC Joint Undertaking
under Horizon Europe Grant No. 101194322 (QEC4QEA), co-funded by the Polish National Centre for Research and
Development (NCBiR) under Decision No. DWM/EuroHPC/2023/429/2025.

## Use of AI tools

We used the AI coding assistant Claude Code (Anthropic; version 2.1.308, with Claude Opus models, Claude Opus 5.5 in the
final stage) to write and run the analysis code and the plotting scripts in this repository, to check the numbers in
the text against the result files, and to draft and edit parts of the text. All theorems were proposed by K.B., and the
AI assistant helped with their proofs, which the authors verified by reading them line by line and by numerical checks
with code independent of the formulas (`analyses/fisher`). All numbers and figures are produced by the code in this
repository from the result files, and no image in the paper is generated by an AI model. The authors reviewed the
code, reran the analyses and checked every number and statement. The statement in the acknowledgments of the paper
is authoritative.
