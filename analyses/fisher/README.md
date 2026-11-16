# Fisher information of homodyne and heterodyne detection: numerical checks

Checks of the Fisher-information results of the paper (Sec. II, Sec. III A and the appendix on proofs and numerical checks). The Fisher information of
both detections is computed directly from the outcome densities, `int delta^2/p` on a grid (`fisherlib.py`),
without the formulas being checked. Run every script from this directory; results go to `out/*.json`, and the
long scripts are resumable (finished cases are skipped).

Numbering as in the paper. Theorem 1 is the structure theorem (order-n blocks of the Fisher forms as symmetric powers,
Eq. (blocks)), Theorem 2 the bound for one Gaussian setting and Theorem 3 the bound for recorded random settings. The thermal-state
results are Eq. (spectrum) (efficiencies of the Hermite-Gauss modes), Eq. (thermalbounds) (bounds 1/2 for homodyne detection at a
uniformly random phase and lambda for heterodyne detection) and Eq. (criterion) (which detection is better for a mode); Eq. (corhom)
is the bound for homodyne detection with a recorded phase of any distribution.

| script | what it checks | output |
|---|---|---|
| `c1b_verification.py` | Hermite-Gauss mode weights of `F_Q` and `F_W` against direct integrals | `out/c1b.json` |
| `c2_counterexamples.py` | counterexample search (also `d_squeezed_random`: orthogonally squeezed vacua with homodyne at a random recorded phase, the convention of the paper) for Eqs. (criterion) and (thermalbounds) and the efficiencies at coherent states (Fock support up to n = 11); lower bound of `F_W/F_Q`; range of the leading order of the Chernoff exponent (`c_local`, and `c_local_vac` near the vacuum down to eps = 1e-6) | `out/c2.json` |
| `c4_sld.py`, `c4b_bound.py` | efficiencies `F/F_SLD` near thermal states, the bound 1/2 for random-phase homodyne detection | `out/c4.json`, `out/c4b.json` |
| `c5_proof_hom.py` | homodyne bound 1/2 of Eq. (thermalbounds) by a direct proof (Meixner-Laguerre identity, angular momentum l = 0 and l != 0) | `out/c5.json` |
| `c6_het_sectors.py` | heterodyne Fisher information per phase sector as a generalized eigenproblem (heterodyne bound of Eq. (thermalbounds)) | `out/c6.json` |
| `c7_noise.py` | loss, dephasing and Gaussian noise act diagonally on the Hermite-Gauss modes (the heterodyne integral skips grid points where the reference `Q` underflows) | `out/c7.json` |
| `c8_global.py` | effect of dephasing on the Chernoff exponents (Sec. III A, from local to global) | `out/c8.json` |
| `c9_sharpness.py` | sharpness: displacements attain both bounds, and `(a^dag)^l rho0 + h.c.` the heterodyne sector bound | `out/c9.json` |
| `c10_sectors.py`, `c10_diag.py`, `c10_meixner.py` | Eq. (spectrum) (spectrum of the modes near a thermal state): suprema for each angular momentum by a generalized eigenproblem, commutation of the forms, simultaneous diagonalization by the Meixner directions. `c10_diag.py` works in a truncated Fock basis (N = 40); it is exact only for small nbar, and its residual commutator at nbar = 2 and 5 is a truncation effect (at nbar = 5, l = 3: 0.139, 0.068, 0.030 for N = 40, 80, 140). The test at large nbar is `c10_meixner.py` | `out/c10.json`, `out/c10_meixner.json` |
| `c11_spectrum_proof.py`, `c11b.py` | each identity of a direct proof of Eq. (spectrum) (Meixner-Laguerre conditional expectation, single Laguerre-Gauss mode, pure Hermite order, bound 1/2) | `out/c11.json` |
| `c12_sld_modes.py` | F_SLD diagonal in the Hermite-Gauss modes, Eq. (sldmodes), against the Fock-basis formula | `out/c12.json` |
| `c14_chi2_global.py` | the Fisher forms as chi^2 divergences for arbitrary states; closed-form Chernoff exponents of a displaced thermal state; bound on the number of events | `out/c14.json` |
| `symlib.py` | library: order-n blocks of the Fisher forms as symmetric powers, M_n = Sym^n(Lambda), J_n = [Sym^n(G_+) + Sym^n(G_-)]/2 (Theorem 1) | |
| `c16_blocks_vs_grid.py` | Theorem 1 against direct integrals of the outcome densities near a squeezed thermal state, all directions of orders n <= 4 (r = 0.6 is limited by the Fock truncation) | `out/c16.json` |
| `c17_bounds.py` | bounds of Theorems 2 and 3 and of Eq. (corhom) for random settings and mixtures; K equally spaced phases; squeezed thermal state; counterexample to the mean-Lambda bound | `out/c17.json` |
| `c18_dephasing_prediction.py` | Sec. III A: prediction of the theory of Sec. II for the effect of dephasing on the fixed pairs of states of Fig. 7 | `out/c18.json` |
| `c19_dephasing_reference.py` | Sec. III A: origin of the offset between the prediction of c18 and the observed effect (reference state of the pair) and the range of validity; marks the pairs in range | `out/c19.json` |
| `c20_multimode.py` | several modes: spectrum and rank of J_n (mixed, partly pure and pure states, M = 2, 3), Theorem 2 for M modes, spectra of the pure states and of the two-mode squeezed vacuum, efficiencies of the physical directions, loss and fixed phases | `out/c20.json` |
| `c21_tmsv_fock.py` | two-mode statements by direct integrals of the outcome densities in the two-mode Fock basis (no use of the Hermite orders); resumable | `out/c21.json` |
| `tmsvlib.py` | library: order-n blocks for several modes and helpers for the two-mode squeezed vacuum | |
| `fisherlib.py`, `c4lib.py` | shared code (no side effects on import) | |

Dependencies: numpy, scipy, mpmath. `c8_global.py` also needs qsdl and the pair matrices
`analyses/chernoff/out/M24_*.npz` (stored in git; made by `python analyses/chernoff/chernoff.py --M 24`, 24 states per family,
seed 2210; the default `--M 40` gives other files).
The numbers quoted in the paper are generated from `out/*.json` by `paper/scripts/numbers_fisher.py`
(`paper/results/numbers_fisher.tex`), which `reproduce_numbers.sh` checks.
