# Fisher information of homodyne and heterodyne detection: numerical checks

Checks of the Fisher-information results of the revised paper (Sec. II, Sec. III A and Appendix A). The Fisher information of
both detections is computed directly from the outcome densities, `int delta^2/p` on a grid (`fisherlib.py`),
without the formulas being checked. Run every script from this directory; results go to `out/*.json`, and the
long scripts are resumable (finished cases are skipped).

Numbering. The paper was rebuilt around a general theorem for Gaussian reference states. In the current numbering, Theorem 1 is the
structure theorem (order-n blocks as symmetric powers), Theorem 2 the bound for one general-dyne setting, Theorem 3 the bound for
recorded random settings, Corollary 1 the bound for homodyne detection with a phase distribution, Corollary 2 the spectrum of the
modes near a thermal state with the bounds 1/2 and lambda, Corollary 3 coherent states and Corollary 4 squeezed thermal states.
The scripts c1-c13 were written for the thermal case; where their docstrings say "Theorem 1", "Theorem 2", "Theorem 3" or "Lemma 1"
of an earlier version, they refer to the spectrum, the homodyne bound, the heterodyne bound and the l = 0 case, which are now all
parts of Corollary 2. The table below uses the current numbering.

| script | what it checks | output |
|---|---|---|
| `c1b_weryfikacja.py` | Hermite-Gauss mode weights of `F_Q` and `F_W` against direct integrals | `out/c1b.json` |
| `c2_kontrprzyklady.py` | counterexample search (also `d_squeezed_random`: orthogonally squeezed vacua with homodyne at a random recorded phase, the convention of the paper) for the mode criterion, Theorems 2 and 3 and Corollary 1 (Fock support up to n = 11); lower bound of `F_W/F_Q`; range of the leading order of the Chernoff exponent (`c_local`, and `c_local_vac` near the vacuum down to eps = 1e-6) | `out/c2.json`, `out/c2b_lower.json` |
| `c4_sld.py`, `c4b_granica.py` | efficiencies `F/F_SLD` near thermal states, the bound 1/2 for random-phase homodyne detection | `out/c4.json`, `out/c4b.json` |
| `c5_dowod_H1.py` | homodyne bound 1/2 of Corollary 2 by a direct proof (Meixner-Laguerre identity, angular momentum l = 0 and l != 0) | `out/c5.json` |
| `c6_het_sektory.py` | heterodyne Fisher information per phase sector as a generalized eigenproblem (heterodyne bound of Corollary 2) | `out/c6.json` |
| `c7_szum.py` | loss, dephasing and Gaussian noise act diagonally on the Hermite-Gauss modes (the heterodyne integral skips grid points where the reference `Q` underflows) | `out/c7.json` |
| `c8_globalne.py` | effect of dephasing on the Chernoff exponents (Sec. III A, from local to global) | `out/c8.json` |
| `c9_ostrosc.py` | sharpness: displacements attain both bounds, and `(a^dag)^l rho0 + h.c.` the heterodyne sector bound | `out/c9.json` |
| `c10_sektory.py`, `c10_diag.py`, `c10_meixner.py` | Corollary 2 (spectrum of the modes near a thermal state): suprema for each angular momentum by a generalized eigenproblem, commutation of the forms, simultaneous diagonalization by the Meixner directions. `c10_diag.py` works in a truncated Fock basis (N = 40); it is exact only for small nbar, and its residual commutator at nbar = 2 and 5 is a truncation effect (at nbar = 5, l = 3: 0.139, 0.068, 0.030 for N = 40, 80, 140). The test of Theorem 1 at large nbar is `c10_meixner.py` | `out/c10.json`, `out/c10_meixner.json` |
| `c11_dowod_widmo.py`, `c11b.py` | each identity of a direct proof of Corollary 2 (Meixner-Laguerre conditional expectation, single Laguerre-Gauss mode, pure Hermite order, bound 1/2) | `out/c11.json` |
| `c12_sld_mody.py` | F_SLD diagonal in the Hermite-Gauss modes, Eq. (sldmodes), against the Fock-basis formula | `out/c12.json` |
| `c13_zanik.py` | thermalization toward rho0: the modes decay as exp(-(k+l/2)At) and the Fisher information of every detection decays mode by mode, without cross terms (Sec. II H) | `out/c13.json` |
| `c14_chi2_global.py` | the Fisher forms as chi^2 divergences for arbitrary states; closed-form Chernoff exponents of a displaced thermal state; bound on the number of events | `out/c14.json` |
| `c15_squeezed_reference.py` | squeezed thermal reference state: efficiencies and coupling of the thermal modes (direct integrals) | `out/c15.json` |
| `symlib.py` | library: order-n blocks of the Fisher forms as symmetric powers, M_n = Sym^n(Lambda), J_n = [Sym^n(G_+) + Sym^n(G_-)]/2 (Theorem 1) | |
| `c16_blocks_vs_grid.py` | Theorem 1 against direct integrals of the outcome densities near a squeezed thermal state, all directions of orders n <= 4 (r = 0.6 is limited by the Fock truncation) | `out/c16.json` |
| `c17_bounds.py` | bounds of Theorems 2 and 3 and of the homodyne corollary for random settings and mixtures; K equally spaced phases; squeezed thermal state; counterexample to the mean-Lambda bound | `out/c17.json` |
| `fisherlib.py`, `c4lib.py` | shared code (no side effects on import) | |

Dependencies: numpy, scipy, mpmath. `c8_globalne.py` also needs qsdl and the pair matrices
`analyses/chernoff/out/M24_*.npz` (stored in git; made by `python analyses/chernoff/chernoff.py --M 24`, 24 states per family,
seed 2210; the default `--M 40` gives other files).
The numbers quoted in the paper are generated from `out/*.json` by `paper/scripts/numbers_fisher.py`
(`paper/results/numbers_fisher.tex`), which `reproduce_numbers.sh` checks.
