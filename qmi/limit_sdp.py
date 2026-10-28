"""Kwantowa granica dokladnosci identyfikacji klasy z JEDNEJ kopii stanu (rowne priory):
   P*(eta) = max_{Pi_c >= 0, sum Pi_c = I}  (1/7) sum_c Re tr(Pi_c sigma_c(eta))      [SDP, MOSEK]
Nie moze jej przekroczyc zaden klasyfikator oparty na pojedynczej kopii, niezaleznie od pomiaru.
   python -m qmi.limit_sdp sigma.npz limit.json [dim]   (dim: obciecie przestrzeni; domyslnie 64)"""
import sys, json, time
import numpy as np, cvxpy as cp

def solve(sig, dim):
    K = sig.shape[0]; Pi = [cp.Variable((dim, dim), hermitian=True) for _ in range(K)]
    cons = [P >> 0 for P in Pi] + [sum(Pi) == np.eye(dim)]
    obj = cp.Maximize(sum(cp.real(cp.trace(Pi[c] @ sig[c][:dim, :dim])) for c in range(K)) / K)
    pr = cp.Problem(obj, cons); pr.solve(solver=cp.MOSEK)
    return float(pr.value), pr.status

if __name__ == "__main__":
    d = np.load(sys.argv[1], allow_pickle=True); dim = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    out = {}
    for i, e in enumerate(d["etas"]):
        t = time.time(); s = d["sigma"][:, i]
        # po obcieciu do dim renormalizujemy slady (straty obciecia znikome dla dim=64)
        s = np.stack([x[:dim, :dim] / np.trace(x[:dim, :dim]).real for x in s])
        v, st = solve(s, dim); out[f"{e:.2f}"] = {"eta": float(e), "P_opt": v, "status": st}
        print(f"eta={e:.2f} straty={100*(1-e):5.1f}%  P*={v:.4f}  [{st}]  {time.time()-t:.1f}s", flush=True)
        json.dump(out, open(sys.argv[2], "w"), indent=1)
