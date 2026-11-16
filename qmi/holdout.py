"""Parameter hold-out: training without selected parameter ranges, test only on them (generalization, not memorization).
The vacuum has no parameters, so it cannot be held out (it is in both)."""
from qsdl.states import sample_state

def in_holdout(label, st):
    if label == "fock":     return st["n"] in (7, 8)
    if label == "coherent": return st["amp"] >= 2.7
    if label == "thermal":  return st["nbar"] >= 3.4
    if label == "cat":      return st["amp"] >= 2.7
    if label == "gkp":      return st["delta"] >= 0.41
    if label == "binomial": return st["S"] == 2 and st["K"] == 4
    return None   # vacuum

def make_sampler(mode):
    """mode: none | train (without the hold-out) | test (hold-out only)."""
    if mode == "none": return sample_state
    def sampler(label, N, rng):
        while True:
            rho, st = sample_state(label, N, rng); h = in_holdout(label, st)
            if h is None or (mode == "train" and not h) or (mode == "test" and h): return rho, st
    return sampler
