"""Druga architektura: ResNet-18 zamiast CNN v2, ten sam protokol treningu co qmi/clf.py.

Protokol skopiowany 1:1 z qmi.clf.train (nie edytujemy oryginalu): skala = std zbioru treningowego, AdamW lr 3e-4,
batch 64, 25 epok (EPOCHS w qmi/eq/kolejka_eq.py), bez augmentacji, model = najnizsza strata walidacyjna (best.pt),
punkt kontrolny co epoke (ckpt.pt) i wznowienie. Jedyna roznica: konstruktor sieci (make_model).

Architektura "r18": torchvision resnet18 (BasicBlock [2,2,2,2], 64-128-256-512 kanalow, polaczenia rezydualne,
globalny average pooling + jedna warstwa liniowa), stem dopasowany do 128x128 i 1 kanalu: conv 3x3 stride 2 (128 -> 64),
bez max-poolingu. Dziala tez dla sinogramow 36x128 (wyjscie layer4: 3x8, GAP).

  python analyses/arch2/clf_arch2.py train --rep Q --seed 0 --train D/train/shots --val D/val/shots --run analyses/arch2/runs/r18_Q_shots_s0
Dodatkowo: --limit N (tani test na podzbiorze), --epochs."""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torchvision
from qmi.data import load_dir

DEV = "cuda" if torch.cuda.is_available() else "cpu"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def make_model(arch="r18", num_classes=7):
    if arch == "v2":
        from qsdl.cnn.classifier import StateClassifierCNNv2
        return StateClassifierCNNv2(num_classes)
    assert arch == "r18", arch
    m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
    nn.init.kaiming_normal_(m.conv1.weight, mode="fan_out", nonlinearity="relu")   # jak reszta warstw w torchvision
    m.maxpool = nn.Identity()
    return m


def nparams(m): return sum(p.numel() for p in m.parameters())


def train(a):
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    tr = load_dir(a.train, a.rep); va = load_dir(a.val, a.rep)
    if a.limit:   # tani test: podzbior (deterministyczny)
        r = np.random.default_rng(0)
        for d in (tr, va):
            ix = np.sort(r.choice(len(d["y"]), min(a.limit, len(d["y"])), replace=False))
            for k in list(d): d[k] = d[k][ix]
    scale = float(tr["X"].std()); log("scale", scale, "n_train", len(tr["y"]), "n_val", len(va["y"]), "shape", tr["X"].shape[1:])
    if not math.isfinite(scale) or scale <= 0: sys.exit(f"BLAD: zla skala {scale}")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = make_model(a.arch).to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    log("arch", a.arch, "parametry", nparams(model))
    Xt = torch.from_numpy(tr["X"] / scale).unsqueeze(1).to(DEV); yt = torch.from_numpy(tr["y"]).to(DEV)
    Xv = torch.from_numpy(va["X"] / scale).unsqueeze(1).to(DEV); yv = torch.from_numpy(va["y"]).to(DEV)
    ck = run / "ckpt.pt"; hist, ep0, best = [], 0, float("inf")
    g = torch.Generator().manual_seed(a.seed)
    if ck.exists():                                                        # wznowienie
        s = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); hist, ep0, best = s["hist"], s["epoch"], s["best"]
        if "gen" in s: g.set_state(s["gen"].cpu())
        else:   # zgodnosc z qmi.clf: tam generator startuje od nowa po wznowieniu; tu odtwarzamy jego stan
            for _ in range(ep0): torch.randperm(len(yt), generator=g)
        log("wznowiono od epoki", ep0)
    crit = nn.CrossEntropyLoss()
    for ep in range(ep0 + 1, a.epochs + 1):
        t0 = time.time(); model.train(); perm = torch.randperm(len(yt), generator=g).to(DEV); tl = tc = 0
        for i in range(0, len(perm), a.bs):
            ix = perm[i:i + a.bs]; opt.zero_grad(); p = model(Xt[ix]); l = crit(p, yt[ix]); l.backward(); opt.step()
            tl += l.item() * len(ix); tc += (p.argmax(1) == yt[ix]).sum().item()
        if not math.isfinite(tl): sys.exit(f"BLAD: strata nieskonczona w epoce {ep}")
        model.eval(); vl = vc = 0
        with torch.no_grad():
            for i in range(0, len(yv), 128):
                p = model(Xv[i:i + 128]); vl += crit(p, yv[i:i + 128]).item() * len(p); vc += (p.argmax(1) == yv[i:i + 128]).sum().item()
        vl /= len(yv); h = dict(epoch=ep, tl=tl / len(yt), ta=tc / len(yt), vl=vl, va=vc / len(yv), sec=round(time.time() - t0, 1)); hist.append(h); log(h)
        if vl < best: best = vl; torch.save({"model": model.state_dict(), "scale": scale, "rep": a.rep, "arch": a.arch, "epoch": ep}, run / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "hist": hist, "epoch": ep, "best": best, "gen": g.get_state()}, str(ck) + ".tmp")
        Path(str(ck) + ".tmp").replace(ck)
    json.dump(hist, open(run / "history.json", "w")); log("TRAIN_DONE", run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--rep", choices=("W", "Q"), required=True); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--train", required=True); t.add_argument("--val", required=True); t.add_argument("--run", required=True)
    t.add_argument("--epochs", type=int, default=25); t.add_argument("--bs", type=int, default=64)
    t.add_argument("--arch", default="r18", choices=("r18", "v2")); t.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); train(a)
