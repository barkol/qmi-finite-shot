"""qmi: finite-record identification of optical quantum states (code of the paper).

Phase-space window. The released results use the grid [-8, 8]^2 with 128 pixels, selected by the environment variable
QMI_XMAX=8. The pinned qsdl commit (a110562) fixes XMAX = 5.0 in qsdl.config, so we set it here, before any qsdl module
that copies XMAX at import time is loaded. Run the code as `python -m qmi.<module>` so that this package is imported first.
"""
import os as _os
import sys as _sys

if "QMI_XMAX" in _os.environ:
    import qsdl.config as _cfg
    _x = float(_os.environ["QMI_XMAX"])
    if _cfg.XMAX != _x:
        _late = [m for m, mod in list(_sys.modules.items())
                 if m.startswith("qsdl.") and m != "qsdl.config" and getattr(mod, "XMAX", _x) != _x]
        if _late:
            raise RuntimeError(f"qsdl modules {_late} were imported with XMAX={_cfg.XMAX} before qmi; import qmi first")
        _cfg.XMAX = _x
