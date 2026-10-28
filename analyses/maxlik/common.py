"""wspolne: modele, obrazy W z rho (GPU), predykcja"""
import numpy as np, torch
from qmi.eval_shots import load_models, pred
from mlcore import GRID

def wimg(rho, Bt, dev="cuda", bs=256):
    """rho (n,D,D) -> W (n,GRID,GRID) float32; Bt = torch (D*D, GRID*GRID) complex64 na dev"""
    out = []
    for i in range(0, len(rho), bs):
        r = torch.as_tensor(np.transpose(rho[i:i + bs], (0, 2, 1)).reshape(len(rho[i:i + bs]), -1), dtype=torch.complex64, device=dev)
        out.append((r @ Bt).real.reshape(-1, GRID, GRID).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def predict_all(models, X, bs=350):
    return {n: np.concatenate([pred(m, X[i:i + bs], sc) for i in range(0, len(X), bs)]) for n, (m, rep, sc) in models.items()}
