"""Data Envelopment Analysis via linear programming (scipy HiGHS).

Envelopment form, for DMU o with inputs x_o (m) and outputs y_o (s):

Input-oriented:   min theta  s.t.  X lam <= theta x_o,  Y lam >= y_o,  lam >= 0
Output-oriented:  max phi    s.t.  X lam <= x_o,        Y lam >= phi y_o
CRS (CCR, Charnes-Cooper-Rhodes 1978) or VRS (BCC, Banker-Charnes-Cooper 1984: sum lam = 1).
Output-oriented efficiency is reported as 1/phi so every score lies in (0, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def _solve(X_ref, Y_ref, x0, y0, rts, orientation):
    """Return (efficiency score, lambda vector) for one DMU against a reference set.

    X_ref: (n, m) inputs, Y_ref: (n, s) outputs of reference DMUs.
    """
    n, m = X_ref.shape
    s = Y_ref.shape[1]
    # decision vector z = [score, lam_1..lam_n]
    c = np.r_[1.0 if orientation == "input" else -1.0, np.zeros(n)]
    if orientation == "input":
        A_in = np.c_[-x0[:, None], X_ref.T]              # X lam - theta x0 <= 0
        b_in = np.zeros(m)
        A_out = np.c_[np.zeros((s, 1)), -Y_ref.T]         # -Y lam <= -y0
        b_out = -y0
    else:
        A_in = np.c_[np.zeros((m, 1)), X_ref.T]           # X lam <= x0
        b_in = x0
        A_out = np.c_[y0[:, None], -Y_ref.T]              # phi y0 - Y lam <= 0
        b_out = np.zeros(s)
    A_ub, b_ub = np.r_[A_in, A_out], np.r_[b_in, b_out]
    A_eq = b_eq = None
    if rts == "vrs":
        A_eq, b_eq = np.r_[0.0, np.ones(n)][None, :], [1.0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (n + 1), method="highs")
    if res.status != 0:
        return np.nan, np.full(n, np.nan)
    score = res.x[0] if orientation == "input" else 1.0 / res.x[0]
    return float(score), res.x[1:]


def efficiency(X, Y, rts="crs", orientation="input", names=None, X_ref=None, Y_ref=None) -> pd.DataFrame:
    """Efficiency score of every DMU (rows of X / Y).

    X, Y: array-like (n, m) and (n, s), or DataFrames. Reference technology defaults to the DMUs themselves.
    Returns DataFrame with 'score' and 'peers' (reference DMUs with lambda > 1e-6).
    """
    if rts not in ("crs", "vrs") or orientation not in ("input", "output"):
        raise ValueError("rts must be 'crs'/'vrs', orientation 'input'/'output'")
    idx = names if names is not None else (X.index if isinstance(X, pd.DataFrame) else range(len(X)))
    Xa, Ya = np.asarray(X, float), np.asarray(Y, float)
    if Xa.ndim == 1: Xa = Xa[:, None]
    if Ya.ndim == 1: Ya = Ya[:, None]
    Xr = Xa if X_ref is None else np.asarray(X_ref, float).reshape(len(X_ref), -1)
    Yr = Ya if Y_ref is None else np.asarray(Y_ref, float).reshape(len(Y_ref), -1)
    ref_names = list(idx) if X_ref is None else list(getattr(X_ref, "index", range(len(Xr))))
    scores, peers = [], []
    for i in range(len(Xa)):
        sc, lam = _solve(Xr, Yr, Xa[i], Ya[i], rts, orientation)
        scores.append(sc)
        peers.append([ref_names[j] for j in np.flatnonzero(lam > 1e-6)] if np.isfinite(sc) else [])
    return pd.DataFrame({"score": scores, "peers": peers}, index=list(idx))


def scale_efficiency(X, Y, orientation="input", names=None) -> pd.DataFrame:
    """CRS, VRS and scale efficiency (SE = CRS / VRS); returns-to-scale classification via NIRS."""
    crs = efficiency(X, Y, "crs", orientation, names)["score"]
    vrs = efficiency(X, Y, "vrs", orientation, names)["score"]
    out = pd.DataFrame({"CRS (technical)": crs, "VRS (pure technical)": vrs, "scale": crs / vrs})
    out["returns to scale"] = np.where(np.isclose(out["scale"], 1, atol=1e-6), "constant", "non-constant")
    return out
