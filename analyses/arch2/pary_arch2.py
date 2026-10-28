"""Sparowana ewaluacja ResNet-18 (r18) i CNN v2 w komorkach reguly, wzorzec: qmi/x8/pary.py.
Te same rekordy co w ewaluacji sparowanej pracy: qmi.eval_shots.gen (K=36, SEED=424242, ziarno = plik, N), komorki
straty 0/50/80 %, defazowanie 20 %, N = 1e2, 1e3, 1e4. Do macierzy dolaczamy sieci v2 z runs_eq (Q, W shots s0-4,
sino s0-4); STRAZNIK: ich poprawnosc stan po stanie musi byc identyczna z runs_eq/pary i runs_eq/pary_sino
(dowod, ze to te same rekordy). Rekordy sa bitowo identyczne (sprawdzone), ale L40S i RTX 5060 Ti zaokraglaja
splot inaczej, wiec przy remisie logitow argmax moze sie przerzucic (diagnoza: tools/diag_v2_remis.py, stan 603 w loss_0 N100,
margines 0.011, CPU float64 zgodny z L40S). Straznik dopuszcza wiec rozbieznosc tylko na stanach z marginesem top-2 < 0.05
i najwyzej 3 na model i komorke; do macierzy wchodza wtedy wiersze v2 z plikow pracy (runs_eq/pary*), a rozbieznosci ida do
<out>/rozbieznosci_v2.json. Wynik: analyses/arch2/pary/<kanal>_<poziom>_N<N>.npz (y, names, correct[model, stan]);
punkt kontrolny = plik na komorke, wznowienie pomija gotowe.
  QMI_XMAX=12 PYTHONPATH=. python analyses/arch2/pary_arch2.py --runs analyses/arch2/runs --workers 12"""
import argparse, os, re, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qmi.eval_shots import gen, DEV
import json
from clf_arch2 import make_model

CELLS = [("loss", 0), ("loss", 50), ("loss", 80), ("dephase", 20)]; NS = (100, 1000, 10000)
V2 = [f"clf_{r}_shots_s{s}" for r in "QW" for s in range(5)] + [f"clf_W_sino_s{s}" for s in range(5)]


def load(p, arch):
    s = torch.load(p, map_location=DEV, weights_only=False); m = make_model(arch).to(DEV)
    m.load_state_dict(s["model"]); m.eval(); return m, s["rep"], s["scale"]


@torch.no_grad()
def logits(m, X, scale):
    return m(torch.from_numpy(X / scale).unsqueeze(1).to(DEV)).double().cpu().numpy()


def gen2(f_N):
    f, N = f_N; img = gen((f, N, 36, "img")); sino = gen((f, N, 36, "sino"))
    if not np.array_equal(img[5], sino[5]): raise RuntimeError(f"rozne etykiety img/sino {f}")
    return img[3], img[4], sino[3], img[5]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="analyses/arch2/runs"); ap.add_argument("--v2", default="runs_eq")
    ap.add_argument("--test", default="data_eq/qmi/test/sweep"); ap.add_argument("--out", default="analyses/arch2/pary")
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--max-cells", type=int, default=99); a = ap.parse_args()
    if os.environ.get("QMI_XMAX") != "12": sys.exit("BLAD: QMI_XMAX musi byc 12 (eq)")
    torch.set_num_threads(2)
    models = {}
    for p in sorted(Path(a.runs).glob("r18_*/best.pt")): models[p.parent.name] = load(p, "r18")
    for n in V2: models[n] = load(Path(a.v2) / n / "best.pt", "v2")
    names = sorted(models); sino = {n for n in names if "_sino_" in n}
    print("modele:", names, flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    ROZ = json.load(open(out / "rozbieznosci_v2.json")) if (out / "rozbieznosci_v2.json").exists() else []
    with Pool(a.workers) as pool:
        for ch, lv in CELLS[:a.max_cells]:
            files = sorted(Path(a.test).glob(f"part_{ch}_{lv:03d}_*.npz"))
            if len(files) != 7: sys.exit(f"BLAD: {ch} {lv}: {len(files)} plikow zamiast 7")
            for N in (NS if a.max_cells >= 4 else NS[:1]):
                dst = out / f"{ch}_{lv}_N{N}.npz"
                if dst.exists():
                    old = list(np.load(dst)["names"])
                    if old == names: print("jest", dst.name, flush=True); continue
                    print("przeliczam (inny zestaw modeli)", dst.name, flush=True)
                ROZ = [r for r in ROZ if r["cell"] != dst.name]
                ys, C, G = [], {n: [] for n in names}, {n: [] for n in names}
                for Xw, Xq, Xs, y in pool.map(gen2, [(str(f), N) for f in files]):
                    ys.append(y)
                    for n in names:
                        m, rep, sc = models[n]; lo = logits(m, Xs if n in sino else (Xw if rep == "W" else Xq), sc)
                        C[n].append(lo.argmax(1) == y); t = np.sort(lo, 1); G[n].append(t[:, -1] - t[:, -2])
                y = np.concatenate(ys); M = np.stack([np.concatenate(C[n]) for n in names])
                for refdir in ("pary", "pary_sino"):   # straznik: v2 zgodne stan po stanie z plikami pracy (poza remisami)
                    r = np.load(Path(a.v2) / refdir / f"{ch}_{lv}_N{N}.npz")
                    if not np.array_equal(r["y"], y): sys.exit(f"BLAD: inne stany niz w {refdir} {dst.name}")
                    for j, n in enumerate(r["names"]):
                        n = str(n); i = names.index(n); bad = np.where(r["correct"][j] != M[i])[0]
                        if len(bad):
                            mg = np.concatenate(G[n])[bad]
                            if len(bad) > 3 or mg.max() >= 0.05: sys.exit(f"BLAD zgodnosci {n} {dst.name}: stany {bad.tolist()} marginesy {mg.tolist()}")
                            ROZ.append(dict(cell=dst.name, model=n, states=bad.tolist(), margins=mg.tolist(), gpu_correct=M[i][bad].tolist()))
                            print("remis", ROZ[-1], flush=True); M[i] = r["correct"][j]
                json.dump(ROZ, open(out / "rozbieznosci_v2.json", "w"), indent=1)
                np.savez(str(dst) + ".tmp.npz", y=y, names=np.array(names), correct=M); os.replace(str(dst) + ".tmp.npz", dst)
                acc = {n: round(float(M[i].mean()), 3) for i, n in enumerate(names) if n.startswith("r18")}
                print(f"{dst.name} zgodne z runs_eq/pary(+sino)  {time.time()-t0:.0f}s  {acc}", flush=True)
    print("PARY_ARCH2_DONE", flush=True)
