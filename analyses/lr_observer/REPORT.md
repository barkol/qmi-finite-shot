# Likelihood-ratio observer for the base set (finite records)

Script: `lr_observer.py` (stages `build`, `run`, `agg`; resumable: the stages write `cache/proto_<cell>.npz` and `parts/part_<cell>_<family>.npz` next to the script, which are regenerated and not stored).
Results: `lr_results.json` (per cell, N and representation: observer error, bootstrap and Wilson 95% intervals, per-family errors, network error with run range, gaps and ratios). Tables below by `make_tables.py`.
Runtime: library build 6 x ~1 min, records and decisions 7 min on 12 threads of a workstation (no GPU).

## Method
* Base set = `qmi/eqn_states.py` (families fock, coherent, sqvac, thermal, cat, gkp, binomial; `<n>` in {2,3,4,5}). Window 12, 128x128 grid (QMI_XMAX=12), noise exactly as in the pipeline (`level_params_eqn`: loss `photonL=1-p`, dephasing `gamma=3p/100`, Gaussian = convolution of W and Q with s2 = p/100; applied to rho, then `qutip.wigner/qfunc`, then convolution).
* Library: Fock 4, thermal 4, GKP 8 (4 `<n>` x mu), binomial 10 (the `_BIN` triples) are finite and exact. Coherent (phase), squeezed vacuum (axis) and cat (phase, 2 parities) have a continuous parameter, so the observer sums over a grid with 2.5 degree steps (144 phases for coherent over 2 pi, 72 for squeezed vacuum and cat over pi). Total 1466 prototypes per cell. Rotations are exact (rho_mn exp(i phi (m-n))), checked against the state constructors to 1e-15. Weights follow the sampling of the base set (uniform over `<n>`, then over the sub-choices, then over the grid).
* Likelihood of N independent events: Q: multinomial over the 16384 pixels with probabilities `clip(Q,0)/sum` (as `heterodyne_image`). W: for each of K phases (K = 36, 12, 6, angles `linspace(0,180,K)`, N split by `split_events`), multinomial over 128 bins of the clipped, normalized discrete Radon marginal (`radon`, bilinear), i.e. the same objects as `homodyne_wigner`. Floors 1e-30 (Q) and 1e-12 (W) avoid log 0. The family score is the log of the weighted sum of prototype likelihoods, the decision is the arg max (equal family prior).
* Records: for each noise cell and family the 100 stored TEST states of `data_eq/qmi/test/sweep` (the same states the networks see), R = 50 records per state and N, so 5000 records per cell, family and N (35000 per cell and N; requirement was 2000). Records are drawn with `numpy.multinomial` from the same exact images, new seeds (not identical records to the networks, same distribution).
* Intervals: percentile bootstrap (4000 resamples) over test states, stratified by family, so the correlation of records from one state is respected. Wilson intervals over records (also in the JSON) ignore that correlation. When the error is 0 in all records the bootstrap is degenerate and the tables show the Wilson upper limit.
* Network errors: 1 - mean pooled accuracy over training runs from `paper/results` (Q and W36: `eval_shots.json` + `eval_shots_s3/s4.json`, 5 runs, keys `a100_*` of numbers.tex; W12: `eval_shots_K12*.json`, 3 runs; W6: `eval_shots_Kscan_K6.json`, 3 runs). The networks were evaluated on loss and dephasing cells only, so Gaussian cells have no network numbers.
* Gap: Q-W in accuracy points = error(W) - error(Q). Ratio = observer gap / network gap, given only where |network gap| >= 1 point.

## Caveats (must go with any use of the numbers)
1. The observer knows the exact prototypes, the noise cell and the window. It is an UPPER BOUND on what any classifier can reach for these known prototypes, and not a performance claim for unknown states (the networks must learn the library from training data and generalize to other states).
2. The families with a continuous parameter are handled with a 2.5 degree grid, so the observer is marginally sub-optimal for them (a state between grid points is explained slightly worse). The errors are therefore upper bounds on the Bayes error. We did not test a finer grid.
3. Finite prototype library means the test states are prototypes of the observer (the same condition as the train/test overlap noted in the Methods of the paper), so near-zero error at N >= 1000 is expected and shows only that these families are separable by likelihood.
4. Records are distributed as those of the networks but are not the same random numbers. Quantization to pixel/bin grid is included. No mode mismatch, no LO drift.
5. Intervals describe Monte Carlo and test-state variation only. The network errors carry run-to-run variation (min and max in the JSON) and the Wilson interval of the pooled 700 test states, which we do not combine with the observer intervals.
6. N=100 cells with error 0.00 have a degenerate bootstrap, see the Wilson upper limits.

## Findings
* At N = 100 the observer error is far below the network error in every cell (noiseless 0.03% vs 4.1% for Q; 20% dephasing 4.5% vs 17.3%; 50% loss 8.2% vs 18.7%; 80% loss 33.1% vs 41.7%). The Chernoff-based pair numbers quoted in OS_G (0.12%, 6.4%, 12.4%, 31.6%, prefactor dropped, pair rather than 7-way) are of the same order as the observer errors, so N = 100 is the regime where the networks, not the data, limit the result.
* The Q-W gap of the observer at N = 100 is at most 1 point in magnitude (noiseless -0.02, dephasing -0.70, 50% loss -0.75, 80% loss +0.37 [-0.27, 1.04]), where the network gap is 4.9, 14.1, 6.5 and 9.8 points. For an ideal observer homodyne with K = 36 phases is not worse than heterodyne at N = 100 (equal within 1 point, W slightly better in three cells). The advantage of Q at N = 100 is therefore a property of the reconstruction plus network pipeline and not of the detector. The ratio observer gap / network gap is -0.00, -0.05, -0.11, +0.04 (K=36).
* At N = 1000 and 80% loss the ideal W observer is better than Q by 3.4 points (5.9% vs 9.3%) and at N = 10000 by 0.8 points (0.15% vs 0.91%), whereas networks give Q-W = +3.5 and -0.3. In Gaussian 50% at N = 100 the observer favors Q (2.7% vs 7.5%, gap +4.8 [4.4, 5.1]), no network data exist for this cell.
* K matters little for the observer (K = 36, 12, 6 within about 0.5 point everywhere), whereas the network error depends on K. The observer is insensitive to K because its likelihood uses all events without reconstruction.
* At N >= 1000 all observer errors are below 0.2%, except 80% loss (5.7-9.3% at N = 1000, 0.14-0.91% at N = 10000).

## Tables (percent error, 95% bootstrap interval; obs = observer, net = mean of stored networks)
]0;▶ cd · | cell | N | Q obs | Q net | W36 obs | W36 net | W12 obs | W12 net | W6 obs | W6 net |
|---|---|---|---|---|---|---|---|---|---|
| noiseless | 100 | 0.03 [0.01, 0.04] | 4.1 | 0.00 [0.00, 0.01] | 9.0 | 0.01 [0.00, 0.02] | 7.0 | 0.00 [0.00, 0.01] | 5.7 |
| noiseless | 1000 | 0.00 [0, 0.01] | 0.1 | 0.00 [0, 0.01] | 0.1 | 0.00 [0, 0.01] | 0.0 | 0.00 [0, 0.01] | 0.2 |
| noiseless | 10000 | 0.00 [0, 0.01] | 0.0 | 0.00 [0, 0.01] | 0.0 | 0.00 [0, 0.01] | 0.0 | 0.00 [0, 0.01] | 0.0 |
| 20% dephasing | 100 | 4.51 [4.19, 4.81] | 17.3 | 3.81 [3.55, 4.07] | 31.4 | 3.65 [3.40, 3.90] | 28.5 | 3.44 [3.18, 3.69] | 25.2 |
| 20% dephasing | 1000 | 0.00 [0, 0.01] | 1.6 | 0.00 [0, 0.01] | 1.4 | 0.00 [0, 0.01] | 1.6 | 0.00 [0, 0.01] | 1.0 |
| 20% dephasing | 10000 | 0.00 [0, 0.01] | 0.1 | 0.00 [0, 0.01] | 0.0 | 0.00 [0, 0.01] | 0.0 | 0.00 [0, 0.01] | 0.0 |
| 50% loss | 100 | 8.16 [7.60, 8.72] | 18.7 | 7.41 [7.01, 7.81] | 25.2 | 7.55 [7.18, 7.91] | 28.5 | 7.03 [6.67, 7.39] | 26.8 |
| 50% loss | 1000 | 0.17 [0.11, 0.23] | 1.5 | 0.01 [0.00, 0.01] | 2.0 | 0.00 [0.00, 0.01] | 2.2 | 0.01 [0.00, 0.01] | 1.7 |
| 50% loss | 10000 | 0.00 [0, 0.01] | 0.0 | 0.00 [0, 0.01] | 0.1 | 0.00 [0, 0.01] | 0.1 | 0.00 [0, 0.01] | 0.0 |
| 80% loss | 100 | 33.07 [32.19, 33.99] | 41.7 | 33.45 [32.68, 34.23] | 51.5 | 33.72 [32.89, 34.55] | 51.8 | 33.34 [32.52, 34.15] | 51.8 |
| 80% loss | 1000 | 9.30 [8.60, 10.05] | 15.4 | 5.86 [5.39, 6.34] | 18.9 | 5.73 [5.26, 6.22] | 18.6 | 5.73 [5.24, 6.23] | 26.5 |
| 80% loss | 10000 | 0.91 [0.75, 1.08] | 4.9 | 0.15 [0.09, 0.21] | 4.6 | 0.15 [0.10, 0.20] | 4.0 | 0.14 [0.09, 0.19] | 6.2 |
| Gaussian 20% (s2=0.2) | 100 | 0.60 [0.50, 0.70] | n/a | 1.11 [0.98, 1.25] | n/a | 1.07 [0.94, 1.19] | n/a | 1.08 [0.96, 1.21] | n/a |
| Gaussian 20% (s2=0.2) | 1000 | 0.00 [0, 0.01] | n/a | 0.00 [0, 0.01] | n/a | 0.00 [0, 0.01] | n/a | 0.00 [0, 0.01] | n/a |
| Gaussian 20% (s2=0.2) | 10000 | 0.00 [0, 0.01] | n/a | 0.00 [0, 0.01] | n/a | 0.00 [0, 0.01] | n/a | 0.00 [0, 0.01] | n/a |
| Gaussian 50% (s2=0.5) | 100 | 2.73 [2.44, 3.02] | n/a | 7.49 [7.12, 7.86] | n/a | 7.27 [6.88, 7.67] | n/a | 6.97 [6.61, 7.36] | n/a |
| Gaussian 50% (s2=0.5) | 1000 | 0.00 [0, 0.01] | n/a | 0.00 [0.00, 0.01] | n/a | 0.00 [0, 0.01] | n/a | 0.01 [0.00, 0.01] | n/a |
| Gaussian 50% (s2=0.5) | 10000 | 0.00 [0, 0.01] | n/a | 0.00 [0, 0.01] | n/a | 0.00 [0, 0.01] | n/a | 0.00 [0, 0.01] | n/a |

| cell | N | gap obs K=36 (pts) | gap net K=36 | ratio | gap obs K=12 | gap net K=12 | ratio | gap obs K=6 | gap net K=6 | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| noiseless | 100 | -0.02 [-0.04, -0.01] | 4.91 | -0.00 | -0.02 [-0.04, 0.00] | 2.99 | -0.01 | -0.02 [-0.04, -0.01] | 1.61 | -0.01 |
| noiseless | 1000 | 0.00 [0.00, 0.00] | 0.00 | - | 0.00 [0.00, 0.00] | -0.01 | - | 0.00 [0.00, 0.00] | 0.18 | - |
| noiseless | 10000 | 0.00 [0.00, 0.00] | 0.00 | - | 0.00 [0.00, 0.00] | 0.00 | - | 0.00 [0.00, 0.00] | 0.00 | - |
| 20% dephasing | 100 | -0.70 [-1.01, -0.37] | 14.09 | -0.05 | -0.86 [-1.17, -0.52] | 11.13 | -0.08 | -1.07 [-1.38, -0.74] | 7.85 | -0.14 |
| 20% dephasing | 1000 | 0.00 [0.00, 0.00] | -0.20 | - | 0.00 [0.00, 0.00] | -0.03 | - | 0.00 [0.00, 0.00] | -0.55 | - |
| 20% dephasing | 10000 | 0.00 [0.00, 0.00] | -0.06 | - | 0.00 [0.00, 0.00] | -0.01 | - | 0.00 [0.00, 0.00] | -0.06 | - |
| 50% loss | 100 | -0.75 [-1.20, -0.30] | 6.51 | -0.11 | -0.61 [-1.10, -0.15] | 9.79 | -0.06 | -1.13 [-1.57, -0.69] | 8.08 | -0.14 |
| 50% loss | 1000 | -0.16 [-0.22, -0.11] | 0.51 | - | -0.17 [-0.23, -0.11] | 0.70 | - | -0.16 [-0.22, -0.11] | 0.23 | - |
| 50% loss | 10000 | 0.00 [0.00, 0.00] | 0.06 | - | 0.00 [0.00, 0.00] | 0.11 | - | 0.00 [0.00, 0.00] | -0.03 | - |
| 80% loss | 100 | 0.37 [-0.27, 1.04] | 9.83 | 0.04 | 0.65 [-0.01, 1.28] | 10.08 | 0.06 | 0.27 [-0.40, 0.91] | 10.12 | 0.03 |
| 80% loss | 1000 | -3.44 [-3.95, -2.95] | 3.51 | -0.98 | -3.57 [-4.05, -3.10] | 3.25 | -1.10 | -3.58 [-4.05, -3.12] | 11.10 | -0.32 |
| 80% loss | 10000 | -0.77 [-0.93, -0.61] | -0.34 | - | -0.77 [-0.92, -0.62] | -0.87 | - | -0.77 [-0.93, -0.62] | 1.28 | -0.60 |

Gap columns: observer gap with bootstrap interval (paired over states), network gap, ratio. "-" means the network gap is below 1 point in magnitude.
