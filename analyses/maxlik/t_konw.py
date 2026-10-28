# test konwencji: tr(Pi rho) vs radon(W obraz danych); baza Wignera vs qt.wigner; ogon rho poza D
import sys, glob, time, numpy as np, qutip as qt
sys.path.insert(0, "analyses/maxlik")
from mlcore import povm, probs, wig_basis, GRID, XMAX
from qmi.shots import radon
TH = np.linspace(0, 180, 36, endpoint=False); D = int(sys.argv[1]) if len(sys.argv) > 1 else 30
for sgn in (1, -1):
    Pi = povm(D, sgn=sgn); tv = []
    for f in sorted(glob.glob("data_eq/qmi/test/sweep/part_loss_000_*.npz")):
        d = np.load(f, allow_pickle=True)
        for i in (0, 37):
            r = d["rho"][i].astype(complex); tail = np.real(np.trace(r) - np.trace(r[:D, :D]))
            p = probs(Pi, r[:D, :D]).reshape(36, GRID); S = radon(d["W"][i].astype(float), TH); S = np.clip(S, 0, None); S /= S.sum(1, keepdims=True)
            tv.append((f.split("_")[-1][:-4], i, round(tail, 5), round(float(0.5 * np.abs(p / p.sum(1, keepdims=True) - S).sum(1).max()), 4)))
    print("sgn", sgn, tv)
t = time.time(); B = wig_basis(D); print("baza", time.time() - t, "s")
d = np.load(sorted(glob.glob("data_eq/qmi/test/sweep/part_loss_000_cat*"))[0], allow_pickle=True)
r = d["rho"][3].astype(complex)[:D, :D]
w1 = np.real(r.reshape(-1) @ B).reshape(GRID, GRID); w2 = np.real(r.T.reshape(-1) @ B).reshape(GRID, GRID)
print("baza vs dane: rho.flat", np.abs(w1 - d["W"][3]).max(), " rho.T.flat", np.abs(w2 - d["W"][3]).max(), " max|W|", np.abs(d["W"][3]).max())
np.save("analyses/maxlik/wbasis_D%d.npy" % D, B.astype(np.complex64))
