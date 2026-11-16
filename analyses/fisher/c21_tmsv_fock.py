"""Direct check of the two-mode statements in the two-mode Fock basis (laboratory frame, no use of the Hermite orders):
F = int delta^2/f for heterodyne detection of both modes (4D grid) and homodyne detection of both modes (2D grid per setting,
settings on a grid), compared with  F/F_SLD = 1/2 + Re <psi, Phi_n conj(psi)>/(2 |psi|^2)  (Eq. (effpsi), Secs. II G and II H
of the paper) and with the closed forms for the directions of the table of Sec. II H and for loss.
Usage: c21_tmsv_fock.py [r]   -> out/c21.json for r = 0.5 (default; about one minute) or out/c21_r<r>.json.
The result file is rewritten after every direction, and a restart skips the directions that are already in it."""
import sys, json, os, time
import numpy as np
from math import factorial
from scipy.special import gammaln
from tmsvlib import symn, tau_local, basis

r = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
big = r > 0.75
NC = 110 if big else 48                       # Fock cutoff per mode
G, RNG = (68, 10.2) if big else (56, 7.0)     # heterodyne grid per real dimension, range in alpha
KS = 40 if big else 32                        # homodyne phases per mode
NX, XR = (400, 17.0) if big else (320, 11.0)
C, S, t = np.cosh(r), np.sinh(r), np.tanh(r)
_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(_D, exist_ok=True)
OUT = os.path.join(_D, 'c21.json' if abs(r - 0.5) < 1e-12 else 'c21_r%.2f.json' % r)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT + '.tmp', 'w'), indent=1); os.replace(OUT + '.tmp', OUT)

sq = np.sqrt(np.arange(1, NC))
def a_(c):  o = np.zeros_like(c); o[:-1, :] = c[1:, :] * sq[:, None]; return o       # a on mode a (first index)
def ad_(c): o = np.zeros_like(c); o[1:, :] = c[:-1, :] * sq[:, None]; return o
def b_(c):  o = np.zeros_like(c); o[:, :-1] = c[:, 1:] * sq[None, :]; return o
def bd_(c): o = np.zeros_like(c); o[:, 1:] = c[:, :-1] * sq[None, :]; return o
def na_(c): return np.arange(NC)[:, None] * c
ip = lambda u, v: np.vdot(u, v)
psi = np.zeros((NC, NC), complex); psi[np.arange(NC), np.arange(NC)] = t ** np.arange(NC) / C
def chi_mode(ma, mb):
    """U|ma, mb> = (C a^dag - S b)^ma (C b^dag - S a)^mb |psi_r> / sqrt(ma! mb!)"""
    c = psi.copy()
    for _ in range(mb): c = C * bd_(c) - S * a_(c)
    for _ in range(ma): c = C * ad_(c) - S * b_(c)
    return c / np.sqrt(factorial(ma) * factorial(mb))

# ---- heterodyne: amplitude <alpha, beta|c> on a 4D grid
g1 = np.linspace(-RNG, RNG, G); dA = (g1[1] - g1[0]) ** 2
al = (g1[:, None] + 1j * g1[None, :]).ravel()
n_ = np.arange(NC)
with np.errstate(divide='ignore', invalid='ignore'):
    logabs = np.where(np.abs(al)[:, None] > 0, n_[None, :] * np.log(np.abs(al))[:, None], np.where(n_[None, :] == 0, 0.0, -np.inf))
Acoh = np.exp(logabs - 0.5 * gammaln(n_ + 1)[None, :] - 0.5 * np.abs(al)[:, None] ** 2) * np.exp(-1j * np.angle(al)[:, None] * n_[None, :])
def amp_het(c): return Acoh @ c @ Acoh.T
def F_het(terms):
    """terms: list of (coef, u, v) for Delta = sum coef |u><v| (Hermitian in total)."""
    a0 = amp_het(psi); f = np.abs(a0) ** 2; del a0
    d = np.zeros(f.shape)
    cache = {}
    for coef, u, v in terms:
        au = amp_het(u); av = au if v is u else amp_het(v)
        d += (coef * au * av.conj()).real; del au, av
    m = f > 1e-280
    return float(np.sum(d[m] ** 2 / f[m]) * dA * dA / np.pi ** 2)

# ---- homodyne: <x_theta|n> = exp(-i n theta) phi_n(x)
xg = np.linspace(-XR, XR, NX); dx = xg[1] - xg[0]
Phi = np.zeros((NC, NX)); Phi[0] = np.pi ** -0.25 * np.exp(-xg ** 2 / 2); Phi[1] = np.sqrt(2) * xg * Phi[0]
for n in range(2, NC): Phi[n] = np.sqrt(2 / n) * xg * Phi[n - 1] - np.sqrt((n - 1) / n) * Phi[n - 2]
def amp_hom(c, tha, thb): return (Phi.T * np.exp(-1j * n_ * tha)[None, :]) @ c @ (Phi * np.exp(-1j * n_ * thb)[:, None])
def F_hom_setting(terms, tha, thb):
    a0 = amp_hom(psi, tha, thb); f = np.abs(a0) ** 2; d = np.zeros(f.shape)
    for coef, u, v in terms:
        au = amp_hom(u, tha, thb); av = au if v is u else amp_hom(v, tha, thb)
        d += (coef * au * av.conj()).real
    m = f > 1e-280
    return float(np.sum(d[m] ** 2 / f[m]) * dx * dx)
def F_hom_uniform(terms, K=KS):
    return float(np.mean([F_hom_setting(terms, np.pi * i / K, np.pi * j / K) for i in range(K) for j in range(K)]))

def regular(chi): return [(1.0, chi, psi), (1.0, psi, chi)]
def T_pred(n, kind, **kw):
    if kind == 'uniform': return symn(tau_local(0, 0, t), n)
    if kind == 'setting': return symn(tau_local(np.exp(2j * kw['tha']), np.exp(2j * kw['thb']), t), n)
def eff_pred(hs, kind, **kw):
    """hs: dict n -> coefficient vector over |m_a, n - m_a> in the order of basis(n, 2)."""
    num = 0; den = 0
    for n, h in hs.items():
        T = T_pred(n, kind, **kw); num += 0.5 * np.vdot(h, h).real + 0.5 * (h.conj() @ T @ h.conj()).real; den += np.vdot(h, h).real
    return float(num / den)

def run(name, fn):
    if name in res: return
    t0 = time.time(); out = fn(); out['seconds'] = time.time() - t0; res[name] = out; save(); print(name, out, flush=True)

rng = np.random.default_rng(11)
res['params'] = dict(r=r, NC=NC, G=G, RNG=RNG, KS=KS, NX=NX, XR=XR,
                     trunc_weight=float(abs(psi[-1, -1]) ** 2), norm_psi=float(ip(psi, psi).real))
# ---- 1. random regular directions of the orders n = 1..4 (and one mixing the orders 1..3)
def rand_dir(orders):
    hs = {}; chi = np.zeros((NC, NC), complex)
    for n in orders:
        b, _ = basis(n, 2); h = rng.normal(size=len(b)) + 1j * rng.normal(size=len(b)); hs[n] = h
        for hm, (ma, mb) in zip(h, b): chi += hm * chi_mode(ma, mb)
    return hs, chi
def test_random(orders):
    def fn():
        hs, chi = rand_dir(orders); nrm = ip(chi, chi).real
        o = {'overlap_with_psi': float(abs(ip(psi, chi))), 'norm2': float(nrm), 'sum_h2': float(sum(np.vdot(h, h).real for h in hs.values()))}
        Fs = 4 * nrm; tm = regular(chi)
        o['het_direct'] = F_het(tm) / Fs; o['het_pred'] = eff_pred(hs, 'uniform')
        o['hom_uniform_direct'] = F_hom_uniform(tm) / Fs; o['hom_uniform_pred'] = o['het_pred']
        o['hom_setting_direct'] = F_hom_setting(tm, 0.3, 1.1) / Fs; o['hom_setting_pred'] = eff_pred(hs, 'setting', tha=0.3, thb=1.1)
        return o
    return fn
for orders in ([1], [2], [3], [4], [1, 2, 3]): run('random_orders_' + ''.join(map(str, orders)), test_random(orders))

# ---- 2. physical directions
nbar = S * S; c2 = np.cosh(2 * r)
def phys_regular(chi, pred_het, pred_FSLD):
    def fn():
        chi_ = chi - ip(psi, chi) * psi; Fs = 4 * ip(chi_, chi_).real; tm = regular(chi_)
        return {'F_SLD': float(Fs), 'F_SLD_pred': float(pred_FSLD), 'het_direct': F_het(tm) / Fs, 'hom_uniform_direct': F_hom_uniform(tm) / Fs, 'pred_both': float(pred_het)}
    return fn
run('squeezing_r', phys_regular(ad_(bd_(psi)) - a_(b_(psi)), (1 + t ** 2) / 2, 4))
run('phase_one_mode', phys_regular(-1j * na_(psi), (1 - t ** 2) / 2, np.sinh(2 * r) ** 2))
w2, w4 = (C * S * (C * C + 3 * S * S)) ** 2, (2 * C * C * S * S) ** 2
run('kerr_one_mode', phys_regular(-1j * na_(na_(psi)), 0.5 - 0.5 * (t ** 2 * w2 + t ** 4 * w4) / (w2 + w4), 4 * (w2 + w4)))
sub = a_(psi); sub = sub / np.sqrt(ip(sub, sub).real); add = ad_(psi); add = add / np.sqrt(ip(add, add).real)
run('coherent_admixture_subtracted', phys_regular(sub, 0.5, 4))
run('coherent_admixture_added', phys_regular(add, 0.5, 4))
def irregular(terms, pq, pw):
    def fn(): return {'F_het_direct': F_het(terms), 'F_het_pred': float(pq), 'F_hom_uniform_direct': F_hom_uniform(terms), 'F_hom_uniform_pred': float(pw), 'F_SLD': 'infinite'}
    return fn
run('chi2_photon_subtracted', irregular([(1.0, sub, sub), (-1.0, psi, psi)], 1.0, 1 + 1 / c2))
run('chi2_photon_added', irregular([(1.0, add, add), (-1.0, psi, psi)], 1.0, 1 + 1 / c2))
apsi = a_(psi); npsi = na_(psi)
run('loss_one_arm', irregular([(1.0, apsi, apsi), (-0.5, npsi, psi), (-0.5, psi, npsi)], nbar / 2, nbar * (1 - 1 / (2 * c2))))
print('all done')
