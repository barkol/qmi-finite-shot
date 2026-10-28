"""Trening i ewaluacja klasyfikatora (Wigner lub Husimi Q) z wznawianiem i przedzialami ufnosci.

  python -m qmi.clf train --rep W --seed 0 --train data/qmi/train/mixed --val data/qmi/val/mixed --run runs/clf_W_s0
  python -m qmi.clf eval  --run runs/clf_W_s0 --test data/qmi/test/sweep
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from qsdl.cnn.classifier import StateClassifierCNNv2
from qmi.data import load_dir, CH_NAMES, wilson

DEV = "cuda" if torch.cuda.is_available() else "cpu"
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


@torch.no_grad()
def predict(model, X, scale, bs=256):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs] / scale).unsqueeze(1).to(DEV)
        out.append(model(x).argmax(1).cpu())
    return torch.cat(out).numpy()


def train(a):
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    tr = load_dir(a.train, a.rep); va = load_dir(a.val, a.rep)
    scale = float(tr["X"].std()); log("scale", scale, "n_train", len(tr["y"]), "n_val", len(va["y"]))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = StateClassifierCNNv2().to(DEV); opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    Xt = torch.from_numpy(tr["X"] / scale).unsqueeze(1).to(DEV); yt = torch.from_numpy(tr["y"]).to(DEV)
    Xv = torch.from_numpy(va["X"] / scale).unsqueeze(1).to(DEV); yv = torch.from_numpy(va["y"]).to(DEV)
    ck = run / "ckpt.pt"; hist, ep0, best = [], 0, float("inf")
    if ck.exists():                                                        # wznowienie
        s = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); hist, ep0, best = s["hist"], s["epoch"], s["best"]
        log("wznowiono od epoki", ep0)
    crit = nn.CrossEntropyLoss(); g = torch.Generator().manual_seed(a.seed)
    for ep in range(ep0 + 1, a.epochs + 1):
        model.train(); perm = torch.randperm(len(yt), generator=g).to(DEV); tl = tc = 0
        for i in range(0, len(perm), a.bs):
            ix = perm[i:i + a.bs]; opt.zero_grad(); p = model(Xt[ix]); l = crit(p, yt[ix]); l.backward(); opt.step()
            tl += l.item() * len(ix); tc += (p.argmax(1) == yt[ix]).sum().item()
        model.eval(); vl = vc = 0
        with torch.no_grad():
            for i in range(0, len(yv), 128):
                p = model(Xv[i:i + 128]); vl += crit(p, yv[i:i + 128]).item() * len(p); vc += (p.argmax(1) == yv[i:i + 128]).sum().item()
        vl /= len(yv); h = dict(epoch=ep, tl=tl / len(yt), ta=tc / len(yt), vl=vl, va=vc / len(yv)); hist.append(h); log(h)
        if vl < best: best = vl; torch.save({"model": model.state_dict(), "scale": scale, "rep": a.rep}, run / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "hist": hist, "epoch": ep, "best": best}, ck)
    json.dump(hist, open(run / "history.json", "w"))


def evaluate(a):
    run = Path(a.run); s = torch.load(run / "best.pt", map_location=DEV, weights_only=False)
    model = StateClassifierCNNv2().to(DEV); model.load_state_dict(s["model"]); rep, scale = s["rep"], s["scale"]
    res = {}
    for chn in a.channels.split(","):
        d = load_dir(a.test, rep, channels=[chn]); p = predict(model, d["X"], scale); res[chn] = {}
        for L in sorted(set(d["level"].tolist())):
            m = d["level"] == L; k = int((p[m] == d["y"][m]).sum()); n = int(m.sum()); lo, hi = wilson(k, n)
            cm = np.zeros((7, 7), int); np.add.at(cm, (d["y"][m], p[m]), 1)
            res[chn][int(L)] = dict(acc=k / n, lo=lo, hi=hi, n=n, cm=cm.tolist())
        log(chn, {L: round(v["acc"], 3) for L, v in res[chn].items() if L % 20 == 0})
    json.dump(res, open(run / a.out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--rep", choices=("W", "Q"), required=True); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--train", required=True); t.add_argument("--val", required=True); t.add_argument("--run", required=True)
    t.add_argument("--epochs", type=int, default=30); t.add_argument("--bs", type=int, default=64)
    e = sp.add_parser("eval"); e.add_argument("--run", required=True); e.add_argument("--test", required=True); e.add_argument("--channels", default="loss,dephase,gauss"); e.add_argument("--out", default="eval_sweep.json")
    a = ap.parse_args(); train(a) if a.cmd == "train" else evaluate(a)
