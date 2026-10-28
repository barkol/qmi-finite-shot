"""zbieznosc RrhoR i czas: jedna komorka, migawki rho w kilku iteracjach -> logL i trafnosc clf_W_exact"""
import sys, time, json, numpy as np, torch
sys.path.insert(0, "analyses/maxlik")
from mlcore import povm, ml_torch
from common import wimg, predict_all, load_models
cell, D, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); B = int(sys.argv[4]) if len(sys.argv) > 4 else 700
d = np.load(f"analyses/maxlik/ev_base/{cell}.npz"); H = d["H"].reshape(len(d["y"]), -1).astype(np.float32)[:B]; y = d["y"][:B]
Pi = povm(D); Bt = torch.as_tensor(np.load(f"analyses/maxlik/wbasis_D{D}.npy"), device="cuda")
models = load_models("analyses/maxlik/runs", r"re:^clf_W_(exact|shots)_s[0-2]$")
SA = tuple(s for s in (100, 300, 1000, 3000, 10000, 20000, 50000) if s <= iters)
torch.cuda.synchronize(); t = time.time(); rho, ll, hist, snaps = ml_torch(H, Pi, iters, save_at=SA); torch.cuda.synchronize(); dt = time.time() - t
print(f"{cell} D={D} B={B} iters={iters}: {dt:.1f}s ({dt/iters*1e3:.2f} ms/iter)", flush=True)
res = {}
for s, r in snaps.items():
    p = predict_all(models, wimg(r, Bt)); li = hist[s // 100 - 1]
    res[s] = dict(dll_vs_final=float(np.mean(hist[-1] - li)), **{n: float((v == y).mean()) for n, v in p.items()})
    print(s, json.dumps({k: round(v, 4) for k, v in res[s].items()}), flush=True)
bad = ~np.isfinite(rho.reshape(len(rho), -1)).all(1); print("NaN rho:", int(bad.sum()), "klasy", np.unique(y[bad]))
