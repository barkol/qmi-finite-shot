"""Hold-out parametrow: trening bez wybranych zakresow, test tylko na nich (test uogolniania, nie memoryzacji).
Vacuum nie ma parametrow, wiec nie da sie go wylaczyc ze zbioru (jest w obu)."""
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
    """mode: none | train (bez hold-outu) | test (tylko hold-out)."""
    if mode == "none": return sample_state
    def sampler(label, N, rng):
        while True:
            rho, st = sample_state(label, N, rng); h = in_holdout(label, st)
            if h is None or (mode == "train" and not h) or (mode == "test" and h): return rho, st
    return sampler
