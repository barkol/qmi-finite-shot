# Fisher information of homodyne and heterodyne detection: numerical checks

Checks of the Fisher-information results of the revised paper (Sec. IV and Appendix A). The Fisher information of
both detections is computed directly from the outcome densities, `int delta^2/p` on a grid (`fisherlib.py`),
without the formulas being checked. Run every script from this directory; results go to `out/*.json`, and the
long scripts are resumable (finished cases are skipped).

| script | what it checks | output |
|---|---|---|
| `c1b_weryfikacja.py` | Hermite-Gauss mode weights of `F_Q` and `F_W` against direct integrals | `out/c1b.json` |
| `c2_kontrprzyklady.py` | counterexample search for the mode criterion, Theorems 1 and 2 and Corollary 1 (Fock support up to n = 11); lower bound of `F_W/F_Q`; range of the leading order of the Chernoff exponent (`c_local`, and `c_local_vac` near the vacuum down to eps = 1e-6) | `out/c2.json`, `out/c2b_lower.json` |
| `c4_sld.py`, `c4b_granica.py` | efficiencies `F/F_SLD` near thermal states, the bound 1/2 for random-phase homodyne detection | `out/c4.json`, `out/c4b.json` |
| `c5_dowod_H1.py` | steps of the proof of Theorem 1 (Meixner-Laguerre identity, sector l = 0 and l != 0) | `out/c5.json` |
| `c6_het_sektory.py` | heterodyne Fisher information per phase sector as a generalized eigenproblem (Theorem 2) | `out/c6.json` |
| `c7_szum.py` | loss, dephasing and Gaussian noise act diagonally on the Hermite-Gauss modes (the heterodyne integral skips grid points where the reference `Q` underflows) | `out/c7.json` |
| `c8_globalne.py` | effect of dephasing on the Chernoff exponents (Sec. IV, global part) | `out/c8.json` |
| `c9_ostrosc.py` | sharpness: displacements attain both bounds, and `(a^dag)^l rho0 + h.c.` the heterodyne sector bound | `out/c9.json` |
| `fisherlib.py`, `c4lib.py` | shared code (no side effects on import) | |

Dependencies: numpy, scipy, mpmath. `c8_globalne.py` also needs qsdl and the pair matrices
`analyses/chernoff/out/M24_*.npz` (stored in git; made by `python analyses/chernoff/chernoff.py --M 24`, 24 states per family,
seed 2210; the default `--M 40` gives other files).
The numbers quoted in the paper are generated from `out/*.json` by `paper/scripts/numbers_fisher.py`
(`paper/results/numbers_fisher.tex`), which `reproduce_numbers.sh` checks.
