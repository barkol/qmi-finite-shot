"""Averaged states of the families sigma_c = E_theta[rho_{c,theta}] and their images under the loss channel (eta = transmission).
The channel is linear, so sigma_c(eta) = E_eta(sigma_c(1)).   python -m qmi.limit_sigma 2000 out.npz [eqn]
The third argument "eqn" = families with equal <n> (qmi.eqn_states, squeezed vacuum instead of vacuum, <n> cycling over the grid).
Note: for eqn, M should be a multiple of len(NBAR), so that every family has the same weights of <n>."""
import sys
from multiprocessing import Pool
import numpy as np, qutip as qt
from qsdl.config import CUTOFF
from qsdl.states import sample_state
from qsdl.noise import photon_loss

def mean_state(args):
    label, M, seed, fam = args; rng = np.random.default_rng(seed); acc = np.zeros((CUTOFF, CUTOFF), complex)
    if fam == "eqn":
        from qmi.eqn_states import make_sampler_eqn; sampler = make_sampler_eqn("none")
    else: sampler = sample_state
    for _ in range(M):
        rho, _ = sampler(label, CUTOFF, rng); acc += rho.full()
    return acc / M

if __name__ == "__main__":
    M = int(sys.argv[1]); out = sys.argv[2]; etas = np.linspace(0, 1, 21); fam = sys.argv[3] if len(sys.argv) > 3 else "base"
    if fam == "eqn":
        from qmi.eqn_states import EQN_LABELS as LABELS
    else: from qsdl.labels import LABELS
    with Pool(7) as p: sig = p.map(mean_state, [(l, M, 5000 + i, fam) for i, l in enumerate(LABELS)])
    S = np.zeros((7, len(etas), CUTOFF, CUTOFF), complex)
    for c in range(7):
        for i, e in enumerate(etas): S[c, i] = photon_loss(qt.Qobj(sig[c]), float(e)).full()
    np.savez(out, sigma=S, etas=etas, labels=np.array(LABELS), M=M)
    print("written", out, S.shape, "traces:", np.round([np.trace(S[c, 10]).real for c in range(7)], 4))
