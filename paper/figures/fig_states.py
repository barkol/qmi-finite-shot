"""Methods figure replacing tab:states: the seven families of the base set at <n> = 3, built with the training
constructors (qmi.eqn_states / qsdl.states, cutoff 64) with phase and squeezing axis fixed at 0.
Rows: exact Wigner W, exact Husimi Q (qutip, g = sqrt 2, vacuum variance 1/2, display window [-7, 7]^2; training images cover [-12, 12]^2 (QMI_XMAX=12)),
photon-number distribution P(n), n = 0..15.
Data separate from drawing: figures/data/fig_states.npz is built if missing (--recompute forces it);
--replot only reads the cache and fails if it is missing."""
import os, sys

sys.dont_write_bytecode = True                      # no __pycache__ in qsdl-qmi or OneDrive
HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "data", "fig_states.npz")
XMAX, NG, NP = 7.0, 281, 16

FAMS = ["sqvac", "fock", "coherent", "thermal", "cat", "binomial", "gkp"]   # order of the table
NAME = {"sqvac": "squeezed vacuum", "fock": "Fock", "coherent": "coherent", "thermal": "thermal",
        "cat": "cat", "binomial": "binomial", "gkp": "GKP"}
NB = 3


def build():
    import numpy as np
    sys.path.insert(0, os.environ.get("QMI_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # read-only import
    import qutip as qt
    from qsdl.config import CUTOFF
    from qsdl.states import fock_state, coherent_state, thermal_state, cat_state, binomial_state, gkp_state
    from qmi.eqn_states import _trunc_param, _cat_amp, _gkp_delta, _BIN, sqvac_state, nbar
    assert CUTOFF == 64
    par = {}
    r = _trunc_param("sqvac", NB); par["sqvac"] = r
    nt = _trunc_param("thermal", NB); par["thermal"] = nt
    amp = _cat_amp(NB, 0.0); par["cat"] = amp
    S, K, mu = 1, 2, 0; assert (S, K, mu) in _BIN[NB]; par["binomial"] = (S, K, mu)
    dg = _gkp_delta(NB, 0); par["gkp"] = dg
    rho = {"sqvac": sqvac_state(CUTOFF, r, 0.0),
           "fock": fock_state(CUTOFF, NB),
           "coherent": coherent_state(CUTOFF, np.sqrt(NB) + 0j),
           "thermal": thermal_state(CUTOFF, nt),
           "cat": cat_state(CUTOFF, amp + 0j, 0.0),
           "binomial": binomial_state(CUTOFF, S, K, mu),
           "gkp": gkp_state(CUTOFF, dg, 0)}
    x = np.linspace(-XMAX, XMAX, NG)
    W = np.array([qt.wigner(rho[f], x, x) for f in FAMS])
    Q = np.array([qt.qfunc(rho[f], x, x) for f in FAMS])
    diag = np.array([np.real(np.diag(rho[f].full())) for f in FAMS])
    nb = np.array([nbar(rho[f]) for f in FAMS])
    dx = x[1] - x[0]
    return dict(x=x, W=W, Q=Q, P=diag[:, :NP], tail=diag[:, NP:].sum(1), nbar=nb, fams=np.array(FAMS),
                normW=W.sum((1, 2)) * dx * dx, normQ=Q.sum((1, 2)) * dx * dx,
                r_sq=par["sqvac"], nth=par["thermal"], a_cat=par["cat"], a_coh=np.sqrt(NB),
                bin_SKmu=np.array(par["binomial"]), delta_gkp=par["gkp"], cutoff=CUTOFF)


def get():
    import numpy as np
    if os.path.exists(NPZ) and "--recompute" not in sys.argv:
        return dict(np.load(NPZ))
    if "--replot" in sys.argv: raise SystemExit(f"--replot: cache {NPZ} missing")
    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    d = build(); np.savez(NPZ, **d); return dict(np.load(NPZ))


def plot(d):
    sys.path.insert(0, os.environ.get("FIGSTYLE_DIR", os.path.dirname(os.path.abspath(__file__))))   # read-only template
    import figstyle as fs
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, PowerNorm
    from scipy.ndimage import maximum_filter
    fs.apply()
    sys.path.insert(0, HERE)
    from figlook import big2mm                          # smallest capitals/numerals >= 2 mm (9 pt floor, scripts too)
    big2mm(plt)
    FS = 9.0                                            # ticks and condition lines (>= 9 pt -> digits >= 2.04 mm)
    FL = 10.0                                           # row labels W, Q, P(n) and family names
    x = d["x"]; ext = [x[0], x[-1], x[0], x[-1]]
    for i, f in enumerate(FAMS): print(f"{f:9s} <n> = {d['nbar'][i]:.9f}  sum P(n<16) = {1 - d['tail'][i]:.6f}  "
                                       f"int W = {d['normW'][i]:.5f}  int Q = {d['normQ'][i]:.5f}  Wmin = {d['W'][i].min():+.4f}")
    S, K, mu = (int(v) for v in d["bin_SKmu"])
    COND = {"sqvac": (r"$\hat S(r)|0\rangle$", r"$\sinh^2 r=\bar n$", r"$r=%.3f$" % d["r_sq"]),
            "fock": (r"$|n\rangle$", r"$n=\bar n$", r"$n=3$"),
            "coherent": (r"$|\alpha\rangle$", r"$|\alpha|^2=\bar n$", r"$\alpha=%.3f$" % d["a_coh"]),
            "thermal": (r"$\rho_{\mathrm{th}}$", r"mean $\bar n$", r"$\bar n=3$"),
            "cat": (r"$|\alpha\rangle+|{-\alpha}\rangle$", r"$|\alpha|^2\tanh|\alpha|^2$", r"$\alpha=%.3f$" % d["a_cat"], r"${=}\,\bar n$"),
            "binomial": (r"$(|0\rangle{+}\sqrt3\,|4\rangle)/2$", r"$(s{+}1)(J{+}1)$",
                         r"$(s,J,\mu)=(%d,%d,%d)$" % (S, K, mu), r"${=}\,2\bar n$"),
            "gkp": (r"square lattice", r"$\Delta_{\mathrm G}$ from $\langle\hat n\rangle$",
                    r"$\mu{=}0$, $\Delta_{\mathrm G}{=}\,%.3f$" % d["delta_gkp"])}

    W_IN, H_IN = 7.0, 2.72
    fig = plt.figure(figsize=(W_IN, H_IN))
    L, R = 0.50, 0.52                                   # inches: left labels, right colourbars
    pw = 0.72                                           # square panel side (in)
    gap = (W_IN - L - R - 7 * pw) / 6
    hP = 0.30
    yP = 0.84
    yQ = yP + hP + 0.07
    yW = yQ + pw + 0.04
    assert yW + pw < H_IN - 0.02, (yW + pw, H_IN)
    wlim = 1 / np.pi
    QMASK = 0.02
    nQ = PowerNorm(gamma=0.5, vmin=0.0, vmax=1 / (2 * np.pi))   # max Q = 1/(2pi) przy normalizacji z rownania (3)
    cmW, cmQ = plt.get_cmap("RdBu_r"), plt.get_cmap("YlOrBr")
    nW = TwoSlopeNorm(vcenter=0.0, vmin=-wlim, vmax=wlim)
    for i, f in enumerate(FAMS):
        x0 = (L + i * (pw + gap)) / W_IN
        aw = fig.add_axes([x0, yW / H_IN, pw / W_IN, pw / H_IN])
        aq = fig.add_axes([x0, yQ / H_IN, pw / W_IN, pw / H_IN])
        ap = fig.add_axes([x0, yP / H_IN, pw / W_IN, hP / H_IN])
        imW = aw.imshow(d["W"][i], origin="lower", extent=ext, cmap=cmW, norm=nW, interpolation="bilinear", rasterized=True)
        if f not in ("sqvac", "coherent", "thermal"):     # Gaussian: W >= 0 exactly (tiny negatives = cutoff artefact)
            Wm = np.ma.masked_where(maximum_filter(np.abs(d["W"][i]), size=25) < QMASK, d["W"][i])  # drop ~1e-3 sign flips far out
            aw.contour(x, x, Wm, levels=[0.0], colors=fs.INK, linewidths=0.35, linestyles="--")
        imQ = aq.imshow(d["Q"][i], origin="lower", extent=ext, cmap=cmQ, norm=nQ, interpolation="bilinear",
                        rasterized=True)
        for a in (aw, aq):
            for s in a.spines.values(): s.set_visible(True); s.set_linewidth(0.4)
            a.set_xticks([-5, 0, 5]); a.set_yticks([-5, 0, 5])
            a.tick_params(length=1.5, width=0.4, labelsize=FS, pad=1, direction="out")
            a.set_xticklabels([]);
            if i: a.set_yticklabels([])
        if i == 0:
            aw.set_ylabel(r"$p$", fontsize=FS, labelpad=0); aq.set_ylabel(r"$p$", fontsize=FS, labelpad=0)
            xl = 0.10 / W_IN                                 # row labels in a fixed column at the left edge
            fig.text(xl, (yW + pw / 2) / H_IN, r"$W$", fontsize=FL, ha="center", va="center")
            fig.text(xl, (yQ + pw / 2) / H_IN, r"$Q$", fontsize=FL, ha="center", va="center")
            fig.text(xl, (yP + hP / 2) / H_IN, r"$P(n)$", fontsize=FL, ha="center", va="center", rotation=90)
        n = np.arange(NP)
        ap.bar(n, d["P"][i], width=0.75, color="0.45", lw=0)
        ap.axvline(NB, color=fs.ORANGE, lw=0.7, ls=(0, (2.5, 1.5)), zorder=3)
        ap.set_xlim(-0.8, NP - 0.2); ap.set_ylim(0, 1.05)
        ap.set_xticks([0, 3, 8, 15]); ap.set_xticklabels(["$0$", "$3$", "$8$", "$15$"])
        ap.set_yticks([0, 0.5, 1]); ap.set_yticklabels(["$0$", "", "$1$"] if i == 0 else [])   # 0.5 unlabelled: no room at 9 pt
        ap.tick_params(labelsize=FS, length=1.5, width=0.4, pad=1)
        for s in ("left", "bottom"): ap.spines[s].set_linewidth(0.4)
        xc = x0 + pw / W_IN / 2
        a, b = COND[f][:2]                              # wide conditions continue on a fourth line
        fig.text(xc, 0.665 / H_IN, NAME[f], fontsize=FL, ha="center", va="top")
        fig.text(xc, 0.490 / H_IN, a, fontsize=FS, ha="center", va="top")
        fig.text(xc, 0.320 / H_IN, b, fontsize=FS, ha="center", va="top")
        if len(COND[f]) > 3: fig.text(xc, 0.155 / H_IN, COND[f][3], fontsize=FS, ha="center", va="top")
    # shared colourbars (right)
    xcb = (W_IN - R + 0.07) / W_IN; wcb = 0.06 / W_IN
    cw = fig.add_axes([xcb, yW / H_IN, wcb, pw / H_IN]); cq = fig.add_axes([xcb, yQ / H_IN, wcb, 0.85 * pw / H_IN])
    b1 = fig.colorbar(imW, cax=cw, ticks=[-wlim, 0, wlim]); b1.ax.set_yticklabels([r"$-1/\pi$", "$0$", r"$1/\pi$"])
    b2 = fig.colorbar(imQ, cax=cq, ticks=[0, 0.02, 0.05, 1 / (2 * np.pi)]); b2.ax.set_yticklabels(["$0$", "$0.02$", "$0.05$", r"$1/2\pi$"])
    for b in (b1, b2):
        b.ax.tick_params(labelsize=FS, length=1.5, width=0.4, pad=1); b.outline.set_linewidth(0.4)
    for ext_ in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "fig_states." + ext_), dpi=600 if ext_ == "png" else None)
    from PIL import Image
    Image.open(os.path.join(HERE, "fig_states.png")).convert("L").save(os.path.join(HERE, "fig_states_gray.png"))
    print("fig_states.pdf/.png/_gray.png", W_IN, H_IN)


if __name__ == "__main__":
    plot(get())
