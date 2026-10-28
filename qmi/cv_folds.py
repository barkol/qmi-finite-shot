"""5-krotna walidacja krzyzowa sieci: laczy pliki czesciowe zbioru treningowego i walidacyjnego, dzieli je na
5 rozlacznych czesci (warstwowo wg klasy; jednostka podzialu = plik, wiec kopie tego samego stanu w danych 'shots' nie trafiaja
do dwoch czesci) i dla kazdej czesci k tworzy katalogi dowiazan fold{k}/train (pozostale 4 czesci) i fold{k}/val (czesc k).
Podzial zalezy tylko od listy plikow i ziarna (deterministyczny); 'shots' nie ma plikow kanalu gauss, wiec jego podzial jest osobny. Test zostaje staly (data/qmi/test/sweep).
   python -m qmi.cv_folds --train data/qmi/train/mixed --val data/qmi/val/mixed --dst data/cv5/mixed"""
import argparse, json, re
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--val", required=True)
    ap.add_argument("--dst", required=True); ap.add_argument("--k", type=int, default=5); ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    files = [(p, "t") for p in sorted(Path(a.train).glob("part_*.npz"))] + [(p, "v") for p in sorted(Path(a.val).glob("part_*.npz"))]
    byclass = {}
    for p, src in files:
        c = int(re.match(r"part_(\d+)_", p.name).group(1)); byclass.setdefault(c, []).append((p, src))
    rng = np.random.default_rng(a.seed); fold_of = {}
    for c in sorted(byclass):
        lst = sorted(byclass[c], key=lambda t: (t[1], t[0].name)); perm = rng.permutation(len(lst))
        for j, i in enumerate(perm): fold_of[lst[i]] = j % a.k
    dst = Path(a.dst); summary = {}
    for k in range(a.k):
        for part in ("train", "val"):
            d = dst / f"fold{k}" / part; d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("part_*.npz"): old.unlink()
        for (p, src), f in fold_of.items():
            part = "val" if f == k else "train"
            (dst / f"fold{k}" / part / f"part_{src}{p.name[len('part_'):]}").symlink_to(p.resolve())
        summary[k] = {part: len(list((dst / f"fold{k}" / part).glob("part_*.npz"))) for part in ("train", "val")}
    json.dump({"n_files": len(files), "folds": summary, "seed": a.seed}, open(dst / "folds.json", "w"), indent=1)
    print(summary); print("CV_FOLDS_DONE")
