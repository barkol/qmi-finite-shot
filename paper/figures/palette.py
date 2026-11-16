"""One colour palette for all figures of the paper: matplotlib's TURBO.
Categorical roles are fixed positions on the colormap, signed maps use the colormap centred at zero.
hexc(x): hex string for matplotlib; gle(x): 'rgb255(r,g,b)' for GLE scripts. Colour is never the only code:
markers, line styles and filled/open symbols keep the figures readable in grey scale."""
from matplotlib import cm, colors
CMAP = "turbo"
X = {"het": 0.84, "hom": 0.12, "S": 0.45,                       # heterodyne (Q), homodyne (W), network without reconstruction
     "loss": 0.00, "dephase": 0.65, "gauss": 0.30,              # noise channels
     "fock": 0.02, "coherent": 0.70, "thermal": 0.30, "sqvac": 0.50, "cat": 0.12, "gkp": 0.90, "binomial": 0.98, "num": 0.00}
def rgb(x): return tuple(int(round(255 * v)) for v in cm.turbo(float(x))[:3])
def hexc(x): return colors.to_hex(cm.turbo(float(x)))
def gle(x): return "rgb255(%d,%d,%d)" % rgb(x)
def role(name, fmt="hex"): return hexc(X[name]) if fmt == "hex" else gle(X[name]) if fmt == "gle" else rgb(X[name])
