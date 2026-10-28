"""MaxLik dla rekordow homodynowych potoku (adaptacja qmi.x8.konno.ml, oryginal bez zmian).
Model pomiaru: K katow z linspace(0,180,K), na kazdym histogram zdarzen w binach pikseli siatki potoku (GRID binow
o srodkach -XMAX..XMAX). Element POVM: Pi[k,j] = int_bin_j |x_th><x_th| dx, <x_th|n> = exp(-i sgn n th) psi_n(x), hbar=1
(konwencja qt.wigner g=sqrt(2): proznia ma wariancje 1/2). Iteracje R rho R (Hradil/Lvovsky) wsadowo na GPU (torch, complex64)
lub na CPU (numpy, complex128). Obraz W z rho: W = Re sum_ab rho_ab W_ba, baza W_ba = qt.wigner(|b><a|) na siatce potoku."""
import os
from math import factorial, pi
import numpy as np
from numpy.polynomial.hermite import hermval
import qmi  # noqa: F401  sets qsdl.config.XMAX from QMI_XMAX (pinned qsdl has XMAX=5.0); must precede qsdl imports
from qsdl.config import GRID, XMAX

DX = 2 * XMAX / (GRID - 1); EDGES = np.linspace(-XMAX - DX / 2, XMAX + DX / 2, GRID + 1)


def povm(D, K=36, sgn=+1, sub=8):
    """(K*GRID, D, D) complex128; Pi[k][a,b] = <a|x><x|b> scalkowane po binie."""
    th = np.linspace(0, 180, K, endpoint=False)
    xs = (EDGES[:-1, None] + (np.arange(sub)[None, :] + 0.5) / sub * DX).ravel()
    psi = np.stack([np.exp(-xs ** 2 / 2) * hermval(xs, [0] * n + [1]) / np.sqrt(2 ** n * factorial(n) * np.sqrt(pi)) for n in range(D)], 1)
    out = []
    for t in th:
        v = psi * np.exp(-1j * sgn * np.arange(D) * np.deg2rad(t))[None, :]
        out.append(np.einsum("xa,xb->xab", v.conj(), v).reshape(GRID, sub, D, D).sum(1) * DX / sub)
    return np.concatenate(out)


def probs(Pi, rho):
    """p_k = tr(Pi_k rho); rho (D,D) albo (B,D,D)."""
    return np.einsum("kab,...ba->...k", Pi, rho).real


def wig_basis(D):
    """(D*D, GRID*GRID) complex: B[(a,b)] = W(|b><a|), wiec W(rho) = Re(rho.T.reshape(-1) @ B) (sprawdzone z qt.wigner w t_konw.py)."""
    import qutip as qt
    y = np.linspace(-XMAX, XMAX, GRID); B = np.zeros((D, D, GRID * GRID), complex)
    for a in range(D):
        for b in range(a, D):
            if a == b:
                B[a, a] = qt.wigner(qt.fock_dm(D, a), y, y).ravel()
            else:   # W(|b><a|) zespolona; z (|b><a| + |a><b|) i i(|b><a| - |a><b|)
                op1 = qt.Qobj(np.outer(np.eye(D)[b], np.eye(D)[a]) + np.outer(np.eye(D)[a], np.eye(D)[b]))
                op2 = qt.Qobj(1j * (np.outer(np.eye(D)[b], np.eye(D)[a]) - np.outer(np.eye(D)[a], np.eye(D)[b])))
                w1 = qt.wigner(op1, y, y).ravel(); w2 = qt.wigner(op2, y, y).ravel()
                B[a, b] = (w1 - 1j * w2) / 2; B[b, a] = (w1 + 1j * w2) / 2   # W(|b><a|), W(|a><b|)
    return B.reshape(D * D, GRID * GRID)


def ml_torch(H, Pi, iters, dev="cuda", rho0=None, tol=None, chk=100, dtype="c64", save_at=()):
    """H: (B, M) zliczenia; Pi: (M, D, D). Zwraca rho (B,D,D) numpy, logL (B,), historia logL co chk iteracji
    (i slownik rho w iteracjach save_at, jesli podane)."""
    import torch
    cd = torch.complex64 if dtype == "c64" else torch.complex128; rd = torch.float32 if dtype == "c64" else torch.float64
    B, M = H.shape; D = Pi.shape[1]
    P = torch.as_tensor(Pi.reshape(M, D * D), dtype=cd, device=dev)               # Pi_flat
    PT = torch.as_tensor(np.transpose(Pi, (0, 2, 1)).reshape(M, D * D), dtype=cd, device=dev)   # do tr(Pi rho) = Pi^T_flat . rho_flat
    h = torch.as_tensor(H, dtype=rd, device=dev); hn = h / h.sum(1, keepdim=True)   # skala h nie zmienia iteracji; normalizacja chroni fp32
    hn = torch.where(hn < 1e-9, torch.zeros_like(hn), hn)   # tylko dla rzutow dokladnych (N=inf); zliczenia maja hn >= 1/N
    rho = (torch.eye(D, dtype=cd, device=dev).expand(B, D, D) / D).clone() if rho0 is None else torch.as_tensor(rho0, dtype=cd, device=dev)
    hist = []; snaps = {}
    for it in range(iters):
        pr = (rho.reshape(B, D * D) @ PT.T).real.clamp_min(1e-12)                  # (B, M); 1e-12 chroni fp32 przed przepelnieniem RrhoR
        w = (hn / pr).to(cd)
        R = (w @ P).reshape(B, D, D)
        rho = R @ rho @ R; rho = (rho + rho.conj().transpose(1, 2)) / 2
        rho = rho / torch.diagonal(rho, dim1=1, dim2=2).real.sum(1)[:, None, None]
        if (it + 1) % chk == 0 or it == iters - 1:
            pr = (rho.reshape(B, D * D) @ PT.T).real.clamp_min(1e-30)
            hist.append((h * torch.log(pr)).sum(1).double().cpu().numpy())
        if it + 1 in save_at: snaps[it + 1] = rho.cpu().numpy()
    if save_at: return rho.cpu().numpy(), hist[-1], np.array(hist), snaps
    return rho.cpu().numpy(), hist[-1], np.array(hist)
