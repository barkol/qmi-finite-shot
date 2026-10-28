# Does Q-W survive a better FBP? (sensitivity check)

Paths (relative to the pilot working directory): fbp2/{code,data,runs,results,logs,pid}. Existing runs/data untouched (read-only use of data_eq, runs_eq).

## Method
- W networks: same protocol as the paper's W shots nets (qmi.clf train, 25 epochs, lr 3e-4, bs 64, seed s, same train/val parent files data_eq/qmi/{train,val}/mixed, copies 2, seed 31337, same N distribution, K=36).
- Records: qmi/shots2.py. The main rng stream (N draw, Q histogram, discarded order-1 marginal draws) is consumed exactly as in the paper, so N sequences and Q records are bit-identical to the paper's. Only the W counts (separate stream rng2, from the order-2 marginals) and the reconstruction differ.
- Check: the five existing Q nets, evaluated on regenerated records, reproduce runs_eq/pary/*.npz state by state (guard in qmi/pary2.py, passed). The paper's 4.9 and 14.1 are reproduced by qmi/analiza2.py.
- Variants: kak2 = order-2 interpolation + Kak-Slaney filter (the variants are set by `QMI_FBP_ORDER=2`, `QMI_FBP_FILTER=kak` in `qmi/shots.py`); hann2 = same + Hann window on the ramp (added by me because kak2 turned out not to be better, see below; chosen on exact-marginal error, not on classification).

## FBP error at N=infinity, K=36, 700 noiseless base-set test states (median / p90 relative L2 vs exact W)
| variant | median | p90 |
|---|---|---|
| paper (order 1, ramp) | 0.286 | 0.529 |
| order 1 + Kak | 0.285 | 0.528 |
| order 2 + ramp | 0.345 | 0.626 |
| kak2 (order 2 + Kak) | 0.344 | 0.626 |
| hann2 (order 2 + Kak + Hann) | 0.264 | 0.460 |
| order 1 + ramp + Hann | 0.272 | 0.450 |

Important: the "better FBP" (order-2 interpolation, Kak-Slaney filter) is NOT better at K=36 on the base set (window +-12): it is worse (0.344 vs 0.286). Its halving of the error was measured at K=72 on the Ahmed set (window +-8). At K=36 the error is dominated by angular undersampling streaks; order-2 interpolation has less low-pass blur and so shows more of them. A subset scan (70 states) shows order 2 + Kak at K=72 gives about 0.07 vs 0.12 for the paper variant, i.e. the gain appears only with enough angles. Only a Hann apodization gives a modest gain at K=36 (8 percent median, 13 percent p90). So no K=36 FBP here is substantially better than the paper's; the test is a sensitivity check, not a proof with a near-exact reconstruction.

## Pilot (1 seed, s0; N=100; Q = mean of 5 existing Q nets; paired states, 700 per cell)
Training end: val acc 0.925 (kak2), 0.929 (hann2); paper W s0 about 0.93.
| cell | paper W (5 seeds) / Q-W | paper W s0 / Q-W | kak2 s0: W / Q-W | hann2 s0: W / Q-W |
|---|---|---|---|---|
| noiseless | 91.0 / 4.9 | 90.4 / 5.5 | 89.9 / 6.1 | 92.4 / 3.5 |
| 20% dephasing | 68.6 / 14.1 | 68.7 / 13.9 | 73.7 / 8.9 | 73.6 / 9.1 |
(Q 95.9 noiseless, 82.7 dephasing.) CI half-widths with one W seed are states-only, about 2.3 to 3.7 points; seed spread (SE 1 to 2 points) is not included.

Reading: under dephasing the gap shrinks from about 14 to about 9 with both variants (same-seed comparison 13.9 to 8.9 / 9.1), i.e. a larger part of the headline gap is reconstruction/regularization, but Q still leads with the states-interval excluding 0. Noiseless: kak2 does not help (6.1), hann2 reduces to 3.5. Not enough seeds to claim more.

## Status
Full run (5 seeds x {kak2, hann2}, all N, four cells) is running on the workstation, about 10 min per net. No rented GPU instance is needed: the workstation alone finishes the full set in about 1 h.
