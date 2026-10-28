"""Atrybucje GradientSHAP dla sieci CNN (Q, W, S) + odniesienie fizyczne (mapa Hellingera heterodyny).

Etapy (kazdy wznawialny, jeden plik npz na komorke w out/; nic poza analyses/shap nie jest zapisywane):
  conv     ustalenie znakow wyrownania (obrot obrazu, przesuniecie sinogramu) na stanach koherentnych -> out/conv.npz
  inputs   wejscia: obrazy dokladne + R realizacji N=100 dla Q, W(FBP), S(sinogram) -> out/inputs_<cell>.npz
  attr     GradientSHAP wzgledem logitu prawdziwej klasy -> out/attr_<cell>_<model>_<inp>_<base>.npz
  hell     mapa Hellingera |sqrt p - sqrt q|^2 dla najblizszego sigma z innej klasy -> out/hell_<cell>.npz
  sanity   kaskadowa randomizacja wag (Adebayo i in. 2018) -> out/sanity_<model>_<inp>_<step>.npz
  summary  metryki -> out/summary.json + out/maps.npz (mapy klasowe wyrownane dla figur)
  python analyses/shap/shap_compute.py --stage all [--small]
"""
import argparse, json, os, sys, time, zlib
from multiprocessing import get_context
from pathlib import Path
import numpy as np
os.environ.setdefault("QMI_XMAX", "12")
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scipy.ndimage import rotate
from scipy.stats import spearmanr, rankdata

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TEST = ROOT / "data_eq/qmi/test/sweep"
RUNS = ROOT / "runs_eq"
FAMS = ("fock", "coherent", "sqvac", "thermal", "cat", "gkp", "binomial")
CELLS = {"clean": ("loss", 0), "deph20": ("dephase", 20)}
K = 36; NREC = 100; G = 128; XM = 12.0
x = np.linspace(-XM, XM, G)
MODELS = {"Q_s0": "clf_Q_shots_s0", "Q_s1": "clf_Q_shots_s1", "W_s0": "clf_W_shots_s0", "W_s1": "clf_W_shots_s1",
          "S_s0": "clf_W_sino_s0", "S_s1": "clf_W_sino_s1", "Qex_s0": "clf_Q_exact_s0", "Wex_s0": "clf_W_exact_s0"}
MOD_REP = {"Q": "Q", "W": "W", "S": "S", "Qex": "Q", "Wex": "W"}
LOG = open(HERE / "shap.log", "a")

def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(v) for v in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def save(f, **kw):
    tmp = f.with_name(f.name + ".tmp.npz"); np.savez_compressed(tmp, **kw); tmp.replace(f)

# ---------------------------------------------------------------- orientacja
def phase_of(fam, meta):
    """Kat obrotu stanu w przestrzeni fazowej (rad) albo None, gdy rodzina nie ma losowej fazy."""
    if isinstance(meta, str):
        import ast; meta = json.loads(meta) if meta.lstrip().startswith("{\"") else ast.literal_eval(meta)
    if fam in ("coherent", "cat"): return float(meta["phase"])
    if fam == "sqvac": return float(meta["psi"])
    return None

def align_img(img, phi, sgn):
    if phi is None: return img
    return rotate(img, sgn * np.degrees(phi), reshape=False, order=1, mode="constant")

def align_sino(s, phi, sgn):
    """Obrot stanu o phi = przesuniecie cykliczne po katach (krok 180/K) z odbiciem osi polozenia przy przejsciu przez 180 st."""
    if phi is None: return s
    sh = int(np.round(sgn * np.degrees(phi) / (180.0 / K)))
    j = np.arange(K) + sh; wrap = np.floor_divide(j, K) % 2 == 1; j = np.mod(j, K)
    out = s[..., j, :].copy(); out[..., wrap, :] = out[..., wrap, ::-1]; return out

def stage_conv():
    f = OUT / "conv.npz"
    if f.exists(): log("jest conv"); return
    from qmi.shots import sinogram_counts
    d = np.load(TEST / "part_loss_000_coherent.npz", allow_pickle=True); n = 12
    ph = [phase_of("coherent", m) for m in d["meta"][:n]]
    res = {}
    for sgn in (+1, -1):
        A = np.array([align_img(d["Q"][i], ph[i], sgn) for i in range(n)])
        res[f"img{sgn}"] = np.mean([np.corrcoef(A[0].ravel(), A[i].ravel())[0, 1] for i in range(1, n)])
        Sx = [sinogram_counts(d["W"][i], 0, None, K) for i in range(n)]
        B = np.array([align_sino(Sx[i], ph[i], sgn) for i in range(n)])
        res[f"sino{sgn}"] = np.mean([np.corrcoef(B[0].ravel(), B[i].ravel())[0, 1] for i in range(1, n)])
    si = +1 if res["img1"] > res["img-1"] else -1; ss = +1 if res["sino1"] > res["sino-1"] else -1
    log("conv: korelacje po wyrownaniu", {k: round(float(v), 3) for k, v in res.items()}, "-> sgn_img", si, "sgn_sino", ss)
    assert max(res["img1"], res["img-1"]) > 0.75 and max(res["sino1"], res["sino-1"]) > 0.75 and abs(res["img1"] - res["img-1"]) > 0.4  # <n> 2..5: rozne promienie, wiec < 1, "wyrownanie nie dziala"
    # kontrola sqvac (os psi) tym samym znakiem
    e = np.load(TEST / "part_loss_000_sqvac.npz", allow_pickle=True)
    A = np.array([align_img(e["Q"][i], phase_of("sqvac", e["meta"][i]), si) for i in range(n)])
    c = np.mean([np.corrcoef(A[0].ravel(), A[i].ravel())[0, 1] for i in range(1, n)])
    B = np.array([e["Q"][i] for i in range(n)]); c0 = np.mean([np.corrcoef(B[0].ravel(), B[i].ravel())[0, 1] for i in range(1, n)])
    log(f"conv: sqvac korelacja przed {c0:.3f}, po {c:.3f}"); assert c > c0
    save(f, sgn_img=si, sgn_sino=ss, **{k: v for k, v in res.items()})

def conv():
    c = np.load(OUT / "conv.npz"); return int(c["sgn_img"]), int(c["sgn_sino"])

# ---------------------------------------------------------------- wejscia
def _gen_one(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    from qmi.shots import heterodyne_image, homodyne_wigner, homodyne_wigner_exact, sinogram_counts
    Q, W, key, R = args
    rng = np.random.default_rng(np.random.SeedSequence([2210, key]))
    out = dict(Qx=Q, Wx=homodyne_wigner_exact(W, K), Sx=sinogram_counts(W, 0, None, K))
    out["Qr"] = np.stack([heterodyne_image(Q, NREC, rng) for _ in range(R)])
    out["Wr"] = np.stack([homodyne_wigner(W, NREC, rng, K) for _ in range(R)])
    out["Sr"] = np.stack([sinogram_counts(W, NREC, rng, K) for _ in range(R)])
    return out

def vacuum_inputs():
    f = OUT / "vacuum.npz"
    if f.exists(): return dict(np.load(f))
    import qutip as qt
    from qsdl.husimi import apply_husimi; from qsdl.wigner import apply_wigner
    from qmi.shots import homodyne_wigner_exact, sinogram_counts
    r = qt.ket2dm(qt.basis(64, 0)); Q = apply_husimi(r, G, XM); W = apply_wigner(r, G, XM)
    v = dict(Q=Q, W=homodyne_wigner_exact(W, K), S=sinogram_counts(W, 0, None, K), Wtrue=W); save(f, **v); return v

def stage_inputs(cell, M, R, workers):
    f = OUT / f"inputs_{cell}.npz"
    if f.exists(): log("jest", f.name); return
    ch, lv = CELLS[cell]; Q, W, lab, phi, metas, fam_of = [], [], [], [], [], []
    for c, fam in enumerate(FAMS):
        d = np.load(TEST / f"part_{ch}_{lv:03d}_{fam}.npz", allow_pickle=True)
        assert int(d["label"][0]) == c and float(d["xmax"]) == XM, (fam, d["label"][0], d["xmax"])
        Q.append(d["Q"][:M]); W.append(d["W"][:M]); lab += [c] * M
        phi += [np.nan if phase_of(fam, m) is None else phase_of(fam, m) for m in d["meta"][:M]]
        metas += [m if isinstance(m, str) else json.dumps(m) for m in d["meta"][:M]]
    Q = np.concatenate(Q); W = np.concatenate(W); lab = np.array(lab); t0 = time.time()
    tasks = [(Q[i], W[i], zlib.crc32(f"{cell}-{i}".encode()), R) for i in range(len(lab))]
    with get_context("fork").Pool(workers) as pool: res = pool.map(_gen_one, tasks, chunksize=4)
    st = {k: np.stack([r[k] for r in res]).astype(np.float32) for k in res[0]}
    for k, v in st.items(): assert np.isfinite(v).all(), k
    save(f, Qtrue=Q, Wtrue=W, label=lab, phi=np.array(phi), meta=np.array(metas), **st)
    log(f"inputs {cell}: {len(lab)} stanow, R={R}, {time.time() - t0:.0f} s", {k: v.shape for k, v in st.items()})

# ---------------------------------------------------------------- modele i atrybucje
def load_model(name, dev):
    import torch
    from qsdl.cnn.classifier import StateClassifierCNNv2
    s = torch.load(RUNS / MODELS[name] / "best.pt", map_location=dev, weights_only=False)
    m = StateClassifierCNNv2().to(dev); m.load_state_dict(s["model"]); m.eval(); return m, float(s["scale"])

def exkey(name):
    """Klucz obrazu dokladnego: Wex trenowana na dokladnym Wignerze (nie FBP), reszta na obrazach z qmi.shots przy N=0."""
    return "Wtrue" if name.startswith("Wex") else MOD_REP[name.split("_")[0]] + "x"

def get_X(inp, name, kind):
    rep = MOD_REP[name.split("_")[0]]
    if kind == "ex": return inp[exkey(name)][:, None]          # (n,1,H,W)
    return inp[rep + "r"]                                         # (n,R,H,W)

def baselines(inp, name, kind, base, lab, rng):
    """Zwraca funkcje c -> tablica baseline (B,H,W)."""
    rep = MOD_REP[name.split("_")[0]]
    if base == "vac":
        v = vacuum_inputs()["Wtrue" if name.startswith("Wex") else rep]; return lambda c: v[None]
    if base == "zero":
        z = np.zeros_like(inp[rep + "x"][0]); return lambda c: z[None]
    pool = inp[exkey(name)] if kind == "ex" else inp[rep + "r"][:, 0]
    sel = {c: rng.choice(np.where(lab != c)[0], 20, replace=False) for c in range(7)}
    return lambda c: pool[sel[c]]

def gshap(model, scale, X, y, basefn, dev, nsamp=50, bs=8, seed=0, mbi=True):
    """mbi=False: oczekiwany gradient E[df/dx] bez czynnika (x - b) (mapa czulosci, niezalezna od wielkosci wejscia)."""
    import torch
    from captum.attr import GradientShap
    gs = GradientShap(model, multiply_by_inputs=mbi); A = np.zeros_like(X, dtype=np.float32); L = np.zeros((len(X), 7), np.float32)
    for c in range(7):
        idx = np.where(y == c)[0]; B = torch.from_numpy(basefn(c) / scale).float().unsqueeze(1).to(dev)
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]; xb = torch.from_numpy(X[j] / scale).float().unsqueeze(1).to(dev)
            torch.manual_seed(seed + int(j[0]))
            a = gs.attribute(xb, baselines=B, target=int(c), n_samples=nsamp, stdevs=0.0)
            A[j] = a[:, 0].detach().cpu().numpy() / 1.0
            with torch.no_grad(): L[j] = model(xb).cpu().numpy()
    return A, L

def stage_attr(cell, names, kinds, bases, dev, nsamp):
    inp = dict(np.load(OUT / f"inputs_{cell}.npz")); lab = inp["label"]
    for name in names:
        model = None
        for kind in kinds:
            if name.startswith(("Qex", "Wex")) and kind != "ex": continue
            for base in bases:
                f = OUT / f"attr_{cell}_{name}_{kind}_{base}.npz"
                if f.exists(): continue
                if model is None: model, scale = load_model(name, dev)
                t0 = time.time(); X = get_X(inp, name, kind); n, R = X.shape[:2]
                rng = np.random.default_rng(zlib.crc32(f"{cell}{name}{kind}{base}".encode()))
                bf = baselines(inp, name, kind, base.rstrip("G"), lab, rng)
                As, Ls = [], []
                for r in range(R):
                    A, L = gshap(model, scale, np.ascontiguousarray(X[:, r]), lab, bf, dev, nsamp, seed=1000 * r, mbi=not base.endswith("G"))
                    As.append(A); Ls.append(L)
                A = np.stack(As, 1); L = np.stack(Ls, 1); assert np.isfinite(A).all()
                # kompletnosc GradientSHAP: suma atrybucji ~ E[f(x)] - E[f(b)] (kontrola przyblizona)
                acc = float((L.argmax(-1) == lab[:, None]).mean())
                save(f, A=A.astype(np.float16), logits=L, label=lab)
                log(f"attr {f.name}: {A.shape}, acc={acc:.3f}, {time.time() - t0:.0f} s")

# ---------------------------------------------------------------- odniesienie fizyczne
def stage_hell(cell):
    f = OUT / f"hell_{cell}.npz"
    if f.exists(): log("jest", f.name); return
    inp = np.load(OUT / f"inputs_{cell}.npz"); Q = inp["Qtrue"].astype(np.float64); lab = inp["label"]
    P = np.clip(Q, 0, None).reshape(len(Q), -1); P /= P.sum(1, keepdims=True); SP = np.sqrt(P)
    BC = SP @ SP.T                                 # wspolczynnik Bhattacharyya; C_Hell = -log BC (s = 1/2)
    H = np.zeros_like(Q, dtype=np.float32); nn = np.zeros(len(Q), int); ch = np.zeros(len(Q))
    for i in range(len(Q)):
        o = np.where(lab != lab[i])[0]; j = o[np.argmax(BC[i, o])]; nn[i] = j; ch[i] = -np.log(BC[i, j])
        H[i] = ((SP[i] - SP[j]) ** 2).reshape(G, G)
    # homodyna: rozklady brzegowe K katow (wiersze dokladnego sinogramu), C_1/2 = -(1/K) sum_k log BC_k
    Sx = np.clip(inp["Sx"].astype(np.float64), 0, None); Sx /= Sx.sum(2, keepdims=True); SS = np.sqrt(Sx)
    LBC = np.mean(np.log(np.maximum(np.einsum("ikx,jkx->ijk", SS, SS), 1e-300)), axis=2)
    HS = np.zeros_like(Sx, dtype=np.float32); nnS = np.zeros(len(Q), int); chS = np.zeros(len(Q))
    for i in range(len(Q)):
        o = np.where(lab != lab[i])[0]; j = o[np.argmax(LBC[i, o])]; nnS[i] = j; chS[i] = -LBC[i, j]
        HS[i] = (SS[i] - SS[j]) ** 2
    save(f, H=H, nn=nn, C_half=ch, HS=HS, nnS=nnS, C_halfS=chS, label=lab)
    log(f"hell {cell}: mediana C_1/2 het = {np.median(ch):.4f}, hom(K={K}) = {np.median(chS):.4f}")

# ---------------------------------------------------------------- sanity (Adebayo)
CASCADE = ["head.5", "head.2", "features.3", "features.2", "features.1", "features.0"]

def randomize(model, upto, seed):
    import torch
    torch.manual_seed(seed); mods = dict(model.named_modules())
    for top in CASCADE[:upto]:
        for m in mods[top].modules():
            if hasattr(m, "reset_parameters"): m.reset_parameters()
            if hasattr(m, "reset_running_stats"): m.reset_running_stats()
    model.eval(); return model

def stage_sanity(names, kinds, dev, per_class, nsamp, G=""):
    inp = dict(np.load(OUT / "inputs_clean.npz")); lab = inp["label"]
    sel = np.concatenate([np.where(lab == c)[0][:per_class] for c in range(7)])
    for name in names:
        for kind in kinds:
            for step in range(len(CASCADE) + 1):
                f = OUT / f"sanity_{name}_{kind}_{step}{G}.npz"
                if f.exists(): continue
                model, scale = load_model(name, dev); model = randomize(model, step, 77 + step)
                X = np.ascontiguousarray(get_X(inp, name, kind)[sel, 0])
                rng = np.random.default_rng(zlib.crc32(f"san{name}{kind}".encode()))
                bf = baselines(inp, name, kind, "other", lab, rng)
                A, L = gshap(model, scale, X, lab[sel], bf, dev, nsamp, mbi=not G)
                save(f, A=A.astype(np.float32), logits=L, sel=sel); log(f"sanity {f.name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True)
    ap.add_argument("--small", action="store_true", help="tani test: 4 stany/klase, R=1, osobny katalog out_small")
    ap.add_argument("--M", type=int, default=40); ap.add_argument("--R", type=int, default=3); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--nsamp", type=int, default=50); a = ap.parse_args()
    if a.small:
        OUT = HERE / "out_small"; a.M, a.R, a.nsamp = 4, 1, 16
    OUT.mkdir(exist_ok=True); import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.set_num_threads(4)
    log("=== start", vars(a), "dev", dev, "OUT", OUT)
    st = a.stage.split(",") if a.stage != "all" else ["conv", "inputs", "attr", "hell", "sanity"]
    if "conv" in st: stage_conv()
    for cell in CELLS:
        if "inputs" in st: stage_inputs(cell, a.M, a.R, a.workers)
    vacuum_inputs()
    for cell in CELLS:
        if "attr" in st:
            stage_attr(cell, ["Q_s0", "W_s0", "S_s0"], ["ex", "rec"], ["other", "vac", "zero"], dev, a.nsamp)
            stage_attr(cell, ["Q_s1", "W_s1", "S_s1"], ["ex", "rec"], ["other"], dev, a.nsamp)
            stage_attr(cell, ["Qex_s0", "Wex_s0"], ["ex"], ["other"], dev, a.nsamp)
            stage_attr(cell, ["Q_s0", "W_s0", "S_s0", "Q_s1", "W_s1", "S_s1"], ["ex", "rec"], ["otherG"], dev, a.nsamp)
        if "hell" in st: stage_hell(cell)
    if "sanity" in st:
        stage_sanity(["Q_s0", "W_s0", "S_s0"], ["rec", "ex"], dev, 10 if not a.small else 2, a.nsamp)
        stage_sanity(["Q_s0", "W_s0", "S_s0"], ["rec", "ex"], dev, 10 if not a.small else 2, a.nsamp, G="_G")
    log("=== COMPUTE_DONE")
