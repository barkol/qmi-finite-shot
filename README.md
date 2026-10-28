# qmi-finite-shot

Companion code and result files for *Homodyne versus heterodyne detection for classifying families of optical quantum states
from finite records of detection events* (F. Grzywa, S. Dolata and K. Bartkiewicz).
Authors of the code and of the paper: see `AUTHORS` and `CITATION.cff`.

License: MIT (see LICENSE).
The dependency Askartan/qsdl has no license file at the pinned commit; its terms are
being settled with its authors, and qsdl is therefore not included in this repository.

## Repository layout

| path | content |
|---|---|
| `qmi/` | code of the paper: generation (`gen.py`, `eqn_states.py`, `ahmed_states.py`), finite records (`shots.py`, `shots_data.py`, `sino_data.py`, drift), training (`clf.py`), evaluation (`eval_shots.py`, `eval_scores.py`, `eval_drift.py`), baselines, quantum limit (`limit_sdp.py`, `limit_sigma.py`), energy control (`energy_control.py`) |
| `qmi/eq/` | the eq run: plan of stages and networks (`plan_eq.py`), queue (`kolejka_eq.py`), final step (`koniec_eq.py`), homodyne networks with and without reconstruction on the same events (`pary_sw.py`), FBP window check (`okno_fbp.py`) |
| `qmi/x8/` | modules used by the eq run: data guard (`guard.py`), paired evaluation (`pary.py`), derived files (`derived.py`), detection section (`detekcja.py`), experimental GKP records (`konno.py`); they accept `QMI_XMAX=8` or `12` |
| `sm/plan.py` | the stage and network plan (seeds, sizes, arguments) from which `qmi/eq/plan_eq.py` derives the plan of the eq run |
| `analyses/` | additional analyses (scripts, reports, small JSON results), see below |
| `paper/results/`, `paper/results_ahm/` | result files of the base set and of the Ahmed set, `numbers*.tex` (LaTeX macros with all numbers of the paper), confidence intervals `ci*.json` |
| `paper/scripts/` | number generators: `make_numbers.py` (base and Ahmed set), `make_numbers_konno.py`, `holm_rule.py`, `kscan_table.py`, `robustness_headline.py`, `numbers_epr.py`, `fbp2_table.py`; `ci_coverage.py` (coverage check of the intervals); `json_close.py` (comparison tool of `reproduce_numbers.sh`) |
| `paper/figures/` | figure scripts (`fig_design.py`, `fig_detection_3d.py`, `fig_rule.py`, `fig_rule_flow.py`, `fig_shots.py`, `fig_states.py`, `fig_sweeps_col.py`), helpers (`common.py`, `figlook.py`, `figstyle.py`), their data caches in `figures/data/` and `fig_rule_flow.tex` (TikZ, generated); the figures themselves are produced by the scripts |
| `reproduce_numbers.sh` | regenerates all number files from the result files, compares them with the committed versions |
| `MANIFEST_LARGE_FILES.md` | large artifacts not stored in git (networks, data, caches) and where they are |

The files `scores.json` (per-state network outputs, 20 MB and 16 MB) are not stored; the paper uses only their
summaries `scores_summary.json`.

### Analyses (`analyses/`)

| directory | content | numbers file |
|---|---|---|
| `chernoff/` | Chernoff exponents of heterodyne and homodyne detection (K=36 phases) and the quantum Chernoff bound, 24 states per class | `numbers_chernoff.tex` |
| `maxlik/` | MaxLik state reconstruction from the same events + classifier, compared with the Q and W networks | `numbers_maxlik.tex` |
| `arch2/` | second architecture (ResNet-18) on the same paired test sets; agreement of the decision rule | `numbers_arch2.tex` |
| `kawasaki/` | photon-subtracted squeezed states of Kawasaki et al. (2024); record-level test of the Konno records | `numbers_kawasaki.tex` |
| `kawasaki_epr/` | traces of the EPR source of Kawasaki et al. (2025): quadratures, records, classification | (text only) |
| `shap/` | GradientShap attribution maps of the Q, W and sinogram networks | (text only) |

Two further directories hold reports, result files and their own code: `lr_observer/` (likelihood-ratio observer of the base set; `lr_observer.py`, `lr_results.json`, `REPORT.md`) and `fbp2/` (sensitivity check with a better FBP: `code/` with the record, pair and analysis modules and the run scripts (they use the data and networks of the base set, see `MANIFEST_LARGE_FILES.md`), `results_full.json`, `fbp_err.json`, `PILOT_REPORT.md`; `paper/scripts/fbp2_table.py` formats the results).

Each directory has its report (`RAPORT.md`, in Polish) and the JSON summaries that the number
generators read (`chernoff/out/podsumowanie_M24.json`, `maxlik/wyniki_maxlik.json`, `arch2/wyniki.json`,
`kawasaki/out/summary.json`, `kawasaki/out_konno/konno_rek.json`). Network weights, event caches and npz
intermediates are listed in `MANIFEST_LARGE_FILES.md`. The generators stop on the first inconsistency (assertions on
the inputs).

## Reproduction

### Numbers (about a minute)

    ./reproduce_numbers.sh            # PY=<python with numpy and scipy>, default ~/anaconda3/bin/python

The script runs, from the result files in the repository,

    cd paper
    python scripts/make_numbers.py                                    # results/numbers.tex, results/ci.json
    RES=results_ahm PFX=ahm_ OUTDIR=results python scripts/make_numbers.py   # results/numbers_ahm.tex, ci_ahm.json
    RES=results python scripts/make_numbers_konno.py                  # results/numbers_konno.tex
    cd ..
    python analyses/chernoff/numbers_chernoff.py                      # paper/results/numbers_chernoff.tex
    python analyses/maxlik/numbers_maxlik.py                          # paper/results/numbers_maxlik.tex
    python analyses/arch2/numbers_arch2.py                            # paper/results/numbers_arch2.tex
    python -I analyses/kawasaki/numbers_kawasaki.py                   # paper/results/numbers_kawasaki.tex
    python analyses/shap/numbers_shap.py                              # paper/results/numbers_shap.tex
    python paper/scripts/holm_rule.py paper/results                   # paper/results/numbers_holm.tex
    python paper/scripts/kscan_table.py                               # paper/results/numbers_kscan.tex, kscan.json

compares every `numbers*.tex` and `ci*.json` with the committed version (byte by byte for `numbers*.tex`, relative tolerance 1e-9 for the JSON files). It stops with exit code 1 if any file differs. All numbers of the paper enter the text through `\RES{key}` macros defined in `paper/results/numbers*.tex`.

### Figures

Figures: the scripts in `paper/figures/` build their caches in `paper/figures/data/` from the result files and plot
from the caches (`--replot` plots only). The style template `paper/figures/figstyle.py` (LaTeX text, Wong palette,
journal column widths) is part of the repository under the MIT license. All text in the figures is at least 2 mm high
in print (2 mm rule; `big2mm` and `TEX_2MM` in `figlook.py`). `fig_states.py` and `fig_detection_3d.py` compute states with qsdl and need it on `PYTHONPATH`;
the cache of `fig_states` (8.9 MB) is not stored. `fig_rule_flow.py` writes `fig_rule_flow.tex` from `ci.json` and
`ci_ahm.json`. The panel of the detection figure in the paper is made by
`python paper/figures/fig_detection_3d.py --cmap=turbo`. All figure scripts were checked to plot from the stored caches
in a clean clone (`--replot`); `fig_rule_flow.tex` is regenerated byte-identical.

### From scratch (days on one GPU)

The datasets are not stored (about 15 GB); they are regenerated deterministically from the seeds in the plan:

    QMI_XMAX=12 python -m qmi.eq.plan_eq                 # prints 31 stages, 81 networks (base set)
    QEQ_SET=ahm QMI_XMAX=12 python -m qmi.eq.plan_eq     # 23 stages, 54 networks (Ahmed set)
    QMI_ROOT=$PWD QMI_XMAX=12 python -m qmi.eq.kolejka_eq          # data, training, evaluations, baselines (resumable)
    QMI_ROOT=$PWD QMI_XMAX=12 python qmi/eq/koniec_eq.py           # derived files, detection section -> results_eq/

`kolejka_eq.py` is resumable (marker files in `data_eq/_markers/`) and stops on the first error. During production
some networks were trained on cloud machines with a job wrapper that is not part of this repository
(`QEQ_TRAIN_RX` selects the networks trained locally); without it every network is trained locally. The quantum
limit (`limit_loss_dim48.json`) comes from `qmi/limit_sdp.py` on a CPU server with MOSEK.

### Experimental data (Dryad, CC0)

* Konno et al., Science 383, 289 (2024), GKP states: https://doi.org/10.5061/dryad.t76hdr86j.
  Put `quad_*deg.npy` in `data/dryad_gkp/` and run
  `QMI_XMAX=12 python -m qmi.x8.konno --data data/dryad_gkp --runs runs_eq --out runs_eq/konno` (see `qmi/eq/kolejka_eq.py`
  for the arguments of the two network sets); results in `paper/results/konno/`.
* Kawasaki et al. (2024), photon-subtracted squeezed states: https://doi.org/10.5061/dryad.9p8cz8wqn
  (expected in `$QMI_ROOT/dane_eksp/kawasaki2024/raw/`, used by `analyses/kawasaki/kawa.py`).
* Kawasaki et al., Nat. Photonics 19, 271 (2025), EPR source: https://doi.org/10.5061/dryad.dbrv15fbq
  (expected in `$QMI_ROOT/dane_eksp/kawasaki2024_epr/raw/bEPR_for_dryad/`, used by `analyses/kawasaki_epr/`).

## Environment

### Installation (for the pipeline and the analyses)

    git clone https://github.com/Askartan/qsdl && git -C qsdl checkout a11056285ad9d684eb09b982b9c3d88841abe77b
    pip install -r requirements.txt
    export PYTHONPATH=$PWD:$PWD/qsdl

qsdl at this commit cannot be installed with pip (flat layout with several top-level packages), and its
`pyproject.toml` asks for qutip>=5.3, whereas the results were produced with qutip 5.2.3; hence the clone and
`PYTHONPATH`. For bitwise reproduction of generated images set `OMP_NUM_THREADS=1`.

**Phase-space window.** The pinned qsdl commit fixes `XMAX = 5.0` in `qsdl/config.py`. All results of the paper use
`QMI_XMAX=12`. `qmi/__init__.py` sets `qsdl.config.XMAX` from this variable before any qsdl module that copies `XMAX`
at import time is loaded (it raises an error if such a module was imported first). Therefore

* run the code as `QMI_XMAX=12 python -m qmi.<module>` (the package is imported first),
* scripts of `analyses/` that use qsdl import `qmi` before `qsdl` for the same reason,
* no change of qsdl is needed. (The production tree used a copy of qsdl whose `config.py` reads `QMI_XMAX`
  itself; both mechanisms give the same window, and `qmi.x8.guard` checks the window key stored in every generated file.)

**Environment variables.**

| variable | meaning | default |
|---|---|---|
| `QMI_XMAX` | half-width of the phase-space window | unset = 5.0 of qsdl; use `12` |
| `QMI_ROOT` | working directory with the large inputs (`data_eq/`, `data_eq_ahm/`, `runs_eq/`, `runs_ahm_eq/`, `results_eq/`, `dane_eksp/`) | repository root (for `qmi/eq/kolejka_eq.py`, `koniec_eq.py`: `~/work/qsdl-qmi`) |
| `QMI_PAPER` | directory written by the number generators and by `koniec_eq.py` | `paper/` of the repository |
| `FIGSTYLE_DIR` | directory of `figstyle.py` | `paper/figures/` |
| `QMI_PY` | interpreter used by the queue | the running interpreter |

The analysis scripts are run from the repository root (as in their docstrings, e.g.
`QMI_XMAX=12 PYTHONPATH=. python analyses/maxlik/run_ml.py ...`) and expect the large inputs there; put symbolic links
(`ln -s <archive>/runs_eq runs_eq`) or set `QMI_ROOT`. These directories are in `.gitignore`.

### Requirements

Python 3.13 with the versions in `requirements.txt` (conda env `ml` of the authors); `reproduce_numbers.sh` needs only
numpy and scipy, and TeX Live with REVTeX 4.2. Hardware used: one RTX 5060 Ti (training), 28-core CPU (data), CPU
server with MOSEK (SDP of the quantum limit).

## Known limitations

See the Discussion of the paper. In particular: simulated training data, tests on experimental records of two
groups; Wigner records are reconstructed by filtered back-projection; the Fock and binomial families are finite sets.
Trained weights are not archived; they are retrained by the scripts from fixed seeds and available from the authors on request (see `MANIFEST_LARGE_FILES.md` for the list of large files).

## Use of AI tools

We used the AI coding assistant Claude Code (Anthropic; version 2.1.308, with Claude Opus models, Claude Opus 5.5 in the
final stage) to write and run the analysis code and the plotting scripts in this repository, to check the numbers in the
text against the result files, and to draft and edit parts of the text. All numbers and figures are produced by the code
in this repository from the result files, and no image in the paper is generated by an AI model. The authors reviewed
the code, reran the analyses and checked every number and statement. The statement in the paper (Methods, "Use of AI
tools") is the same.
