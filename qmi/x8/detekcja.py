"""Sekcja detekcji: okno XM = qsdl.config.XMAX (QMI_XMAX=12), zdarzenia z qmi.shots (split_events),
wspolczynnik prostej homodyny dopasowany z danych. Dane osobno od rysowania:
<out>/calc_detection.npz (licz-jesli-brak), <out>/check_noise.json, <out>/detekcja.json (dopasowania), rysunki <out>/fig_detection_*.pdf.
   QMI_XMAX=12 python -m qmi.x8.detekcja results_eq/detekcja"""
import sys, os, json, numpy as np
from qsdl.config import XMAX
import qmi.shots as S
OUT = sys.argv[1] if len(sys.argv) > 1 else "results_x8/detekcja"; os.makedirs(OUT, exist_ok=True)
G, XM, K = 128, XMAX, 36
x = np.linspace(-XM, XM, G); dx = x[1] - x[0]; A = dx * dx
NPZ = os.path.join(OUT, "calc_detection.npz")
if not os.path.exists(NPZ):
    import qutip as qt
    a = 2.0; dd = 64
    st = {"cat": qt.ket2dm((qt.coherent(dd, a) + qt.coherent(dd, -a)).unit()), "coh": qt.ket2dm(qt.coherent(dd, 1.5 + 0.0j))}
    res = {}; rng = np.random.default_rng(7); Ns = [10**2, 10**3, 10**4, 10**5, 10**6]; show = [10**3, 10**4, 10**5]
    for name, rho in st.items():
        W = np.asarray(qt.wigner(rho, x, x), float); Q = np.asarray(qt.qfunc(rho, x, x), float)
        W /= W.sum() * A; Q /= Q.sum() * A
        Wex = S.homodyne_wigner_exact(W, K).astype(float); s = float((Wex * W).sum() / (W * W).sum()); Wex /= s
        res[name + "_W"] = W; res[name + "_Q"] = Q; res[name + "_Wfbp_exact"] = Wex
        res[name + "_sigW"] = (W**2).sum() * A; res[name + "_sigQ"] = (Q**2).sum() * A
        iq, iw = [], []
        for N in Ns:
            eq, ew = [], []
            for r in range(8):
                Qh = S.heterodyne_image(Q, N, rng); Wh = S.homodyne_wigner(W, N, rng, K) / s
                eq.append(((Qh - Q)**2).sum() * A); ew.append(((Wh - Wex)**2).sum() * A)
                if r == 0 and N in show: res[f"{name}_Q_N{N}"] = Qh; res[f"{name}_W_N{N}"] = Wh
            iq.append(np.mean(eq)); iw.append(np.mean(ew))
        res[name + "_ISE_Q"] = np.array(iq); res[name + "_ISE_W"] = np.array(iw)
        print(name, "ISE_Q*N", np.array(iq) * Ns, "ISE_W*N", np.array(iw) * Ns, flush=True)
    res["Ns"] = np.array(Ns); res["A"] = A; res["L"] = 2 * XM; res["kc"] = np.pi / dx
    np.savez(NPZ + ".tmp.npz", **res); os.replace(NPZ + ".tmp.npz", NPZ)
# --- check_noise (stan gaussowski, wzory analityczne)
CN = os.path.join(OUT, "check_noise.json")
if not os.path.exists(CN):
    X, P = np.meshgrid(x, x); x0, p0 = 1.5, 1.0
    W = np.exp(-((X - x0)**2 + (P - p0)**2)) / np.pi; Q = np.exp(-((X - x0)**2 + (P - p0)**2) / 2) / (2 * np.pi)
    W /= W.sum() * A; Q /= Q.sum() * A; Wex = S.homodyne_wigner_exact(W, K); s = float((Wex * W).sum() / (W * W).sum())
    L = 2 * XM; kc = np.pi / dx; rng = np.random.default_rng(1); out = {}
    for N in [10**3, 10**4, 10**5, 10**6]:
        eq, ew = [], []
        for r in range(12):
            Qh = S.heterodyne_image(Q, N, rng); eq.append(((Qh - Q)**2).sum() * A)
            Wh = S.homodyne_wigner(W, N, rng, K); ew.append((((Wh - Wex) / s)**2).sum() * A)
        out[N] = dict(ISE_Q=float(np.mean(eq)), pred_Q=float((1 - (Q * Q).sum() * A * A) / (N * A)), ISE_W=float(np.mean(ew)),
                      pred_W=float(kc**3 * L / (12 * np.pi * N)), scale=s)
        print(N, out[N], flush=True)
    json.dump(out, open(CN + ".tmp", "w"), indent=1); os.replace(CN + ".tmp", CN)
# --- dopasowania i rysunki
d = np.load(NPZ); Ns = d["Ns"]; A, L, kc = float(d["A"]), float(d["L"]), float(d["kc"])
fitW = float(np.mean([np.mean(d[n + "_ISE_W"] * Ns) / (kc**3 * L / (12 * np.pi)) for n in ("cat", "coh")]))
fitQ = float(np.mean([np.mean(d[n + "_ISE_Q"] * Ns * A) for n in ("cat", "coh")]))
json.dump({"XM": XM, "dx": dx, "A": A, "L": L, "kc": kc, "wsp_homodyna_fit": fitW, "wsp_homodyna_stary_rysunek": 0.51, "ISE_Q_razy_NA": fitQ},
          open(os.path.join(OUT, "detekcja.json"), "w"), indent=1)
print("wspolczynnik homodyny (dopasowanie):", fitW, "heterodyna ISE*N*A:", fitQ)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    plt.rcParams.update({"text.usetex": True}); fig = plt.figure(); fig.text(0.5, 0.5, r"$x$"); fig.savefig(os.devnull, format="pdf"); plt.close(fig)
except Exception: plt.rcParams.update({"text.usetex": False})
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
ext = [-XM, XM, -XM, XM]; BLUE, ORG = "#0072B2", "#E69F00"; TK = [-6, 0, 6]
fig, ax = plt.subplots(2, 4, figsize=(7.0, 3.7), constrained_layout=True)
cols = [("exact", None)] + [(f"$N=10^{int(np.log10(n))}$", n) for n in (10**3, 10**4, 10**5)]
vq = d["cat_Q"].max(); vw = np.abs(d["cat_W"]).max()
for j, (t, n) in enumerate(cols):
    Qi = d["cat_Q"] if n is None else d[f"cat_Q_N{n}"]; Wi = d["cat_Wfbp_exact"] if n is None else d[f"cat_W_N{n}"]
    a = ax[0, j]; im0 = a.imshow(Qi, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vq); a.set_title(t if n else r"$N\to\infty$")
    a = ax[1, j]; im1 = a.imshow(Wi, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vw, vmax=vw)
    for r in (0, 1):
        ax[r, j].set_xticks(TK); ax[r, j].set_yticks(TK if j == 0 else [])
        if r == 0: ax[r, j].set_xticklabels([])
ax[0, 0].set_ylabel(r"$p$ (heterodyne, $Q$)"); ax[1, 0].set_ylabel(r"$p$ (homodyne, $W$)")
for j in range(4): ax[1, j].set_xlabel(r"$x$")
fig.colorbar(im0, ax=ax[0, :], shrink=0.9, pad=0.01, label=r"$Q$ (a.u.)"); fig.colorbar(im1, ax=ax[1, :], shrink=0.9, pad=0.01, label=r"$W$ (a.u.)")
fig.savefig(os.path.join(OUT, "fig_detection_images.pdf")); plt.close(fig)
from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True); n = np.logspace(2, 6, 50)
for nm, mk in (("cat", "o"), ("coh", "s")):
    ax.loglog(Ns, d[nm + "_ISE_Q"], mk, color=ORG, ms=(7 if nm == "coh" else 4), mfc="none" if nm == "coh" else ORG)
    ax.loglog(Ns, d[nm + "_ISE_W"], mk, color=BLUE, ms=(7 if nm == "coh" else 4), mfc="none" if nm == "coh" else BLUE)
ax.loglog(n, 1 / (n * A), "-", color=ORG, lw=0.8); ax.loglog(n, fitW * kc**3 * L / (12 * np.pi * n), "-", color=BLUE, lw=0.8)
h = [Line2D([], [], color=ORG, lw=0.8, label=r"heterodyne $Q$: $1/(NA)$"),
     Line2D([], [], color=BLUE, lw=0.8, label=rf"homodyne $W$: ${fitW:.2f}\,k_c^3L/(12\pi N)$"),
     Line2D([], [], marker="o", ls="", color="k", ms=4, label="cat"), Line2D([], [], marker="s", ls="", color="k", mfc="none", ms=4, label="coherent")]
ax.legend(handles=h, frameon=False, loc="lower left"); ax.set_xlabel(r"recorded events $N$"); ax.set_ylabel(r"squared error $\int(\hat f-f_\infty)^2\,\mathrm{d}^2r$")
fig.savefig(os.path.join(OUT, "fig_detection_noise.pdf")); plt.close(fig); print("DETEKCJA_DONE", OUT)
