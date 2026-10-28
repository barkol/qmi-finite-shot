"""Sparowana ewaluacja Q vs W na wspolnym zbiorze testowym: poprawnosc kazdego stanu dla kazdej sieci.
Rekordy testowe sa deterministyczne (ziarno = plik, N; qmi.eval_shots.gen), wiec wszystkie sieci widza te same rekordy
i roznice Q-W mozna liczyc parami po stanach. Komorki tabeli reguly: straty 0/50/80%, defazowanie 20%, N=1e2/1e3/1e4.
Wynik: <runs>/pary/<kanal>_<poziom>_N<N>.npz (y, nazwy modeli, macierz poprawnosci [model, stan]); punkt kontrolny = plik
na komorke, wznowienie pomija gotowe. Straznik: suma poprawnych musi sie zgadzac z eval_shots*.json co do stanu.
  QMI_XMAX=12 PYTHONPATH=. ~/anaconda3/envs/ml/bin/python -m qmi.x8.pary --runs runs_eq --test data_eq/qmi/test/sweep \
      --ref runs_eq/eval_shots_all_s0-4.json
--sino <regex> dolacza sieci homodynowe bez rekonstrukcji (wejscie: sinogram z qmi.eval_shots.gen w trybie
"sino", te same stany i ta sama kolejnosc co rekordy obrazowe); wynik w katalogu --out (np. pary_sino), zeby nie mieszac
z plikami pary/ uzywanymi w tabeli regul:
  ... -m qmi.x8.pary --runs runs_eq --test data_eq/qmi/test/sweep --match 're:^clf_Q_shots_s\d+$' \
      --sino 're:^clf_W_sino_s\d+$' --out pary_sino --ref runs_eq/eval_shots_all_s0-4.json runs_eq/eval_shots_sino_s0-4.json"""
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np
from qmi.eval_shots import gen, load_models, pred

CELLS = [("loss", 0), ("loss", 50), ("loss", 80), ("dephase", 20)]; NS = (100, 1000, 10000)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--ref", nargs="+", required=True); ap.add_argument("--match", default=r"re:^clf_[WQ]_shots_s\d+$")
    ap.add_argument("--sino", default=None); ap.add_argument("--out", default="pary")
    a = ap.parse_args()
    if os.environ.get("QMI_XMAX") not in ("8", "12"): sys.exit("BLAD: QMI_XMAX musi byc 8 albo 12; inaczej rekordy na siatce +-5")
    ref = {}
    for f in a.ref: ref.update(json.load(open(f)))
    models = load_models(a.runs, a.match); sino = set()
    if a.sino:
        ms = load_models(a.runs, a.sino); sino = set(ms); models.update(ms)
    names = sorted(models); print("modele:", names, "sinogram:", sorted(sino), flush=True)
    if not names or any(n not in ref for n in names): sys.exit(f"BLAD: modele bez odniesienia w --ref: {[n for n in names if n not in ref]}")
    out = Path(a.runs) / a.out; out.mkdir(exist_ok=True); t0 = time.time()
    for ch, lv in CELLS:
        files = sorted(Path(a.test).glob(f"part_{ch}_{lv:03d}_*.npz"))
        if len(files) != 7: sys.exit(f"BLAD: {ch} {lv}: {len(files)} plikow zamiast 7")
        for N in NS:
            dst = out / f"{ch}_{lv}_N{N}.npz"
            if dst.exists(): print("jest", dst.name, flush=True); continue
            ys, C = [], {n: [] for n in names}
            for f in files:
                _, _, _, Xw, Xq, y = gen((str(f), N, 36, "img")); ys.append(y)
                if sino:
                    _, _, _, Xs, _, ys_ = gen((str(f), N, 36, "sino"))
                    if not np.array_equal(ys_, y): sys.exit(f"BLAD: rozne etykiety img/sino w {f.name}")
                for n in names:
                    m, rep, sc = models[n]; C[n].append(pred(m, Xs if n in sino else (Xw if rep == "W" else Xq), sc) == y)
            y = np.concatenate(ys); M = np.stack([np.concatenate(C[n]) for n in names])
            for i, n in enumerate(names):   # straznik: ta sama liczba poprawnych co w eval_shots
                r = ref[n][ch][str(N)][str(lv)]; k = int(round(r["acc"] * r["n"]))
                if r["n"] != len(y) or k != int(M[i].sum()):
                    sys.exit(f"BLAD zgodnosci {n} {ch} {lv} N={N}: {int(M[i].sum())}/{len(y)} wobec {k}/{r['n']}")
            np.savez(str(dst) + ".tmp.npz", y=y, names=np.array(names), correct=M); os.replace(str(dst) + ".tmp.npz", dst)
            print(f"{dst.name}  zgodne z eval_shots  {time.time()-t0:.0f}s", flush=True)
    print("PARY_DONE", flush=True)
