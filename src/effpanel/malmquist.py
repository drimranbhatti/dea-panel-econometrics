"""Malmquist total-factor-productivity index (Fare, Grosskopf, Norris & Zhang, 1994).

    M = EC x TC
    EC (efficiency change)  = D^{t+1}(x_{t+1}, y_{t+1}) / D^t(x_t, y_t)
    TC (technical change)   = sqrt[ D^t(x_{t+1}, y_{t+1}) / D^{t+1}(x_{t+1}, y_{t+1})
                                   x D^t(x_t, y_t)        / D^{t+1}(x_t, y_t) ]
Distances are input-oriented CRS DEA scores; M > 1 means productivity growth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dea import _solve


def _dist(Xr, Yr, x, y):
    return _solve(Xr, Yr, x, y, "crs", "input")[0]


def malmquist(panel: pd.DataFrame, inputs: list[str], outputs: list[str], entity="entity", time="year") -> pd.DataFrame:
    """Malmquist index for consecutive periods. ``panel`` is long format with one row per entity-year.

    Only entities observed in both periods enter each comparison (balanced pairs).
    """
    years = sorted(panel[time].unique())
    rows = []
    for t0, t1 in zip(years[:-1], years[1:]):
        a = panel[panel[time] == t0].set_index(entity)
        b = panel[panel[time] == t1].set_index(entity)
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        X0, Y0, X1, Y1 = (d[c].to_numpy(float) for d, c in ((a, inputs), (a, outputs), (b, inputs), (b, outputs)))
        for i, e in enumerate(common):
            d00 = _dist(X0, Y0, X0[i], Y0[i]); d11 = _dist(X1, Y1, X1[i], Y1[i])
            d01 = _dist(X0, Y0, X1[i], Y1[i]); d10 = _dist(X1, Y1, X0[i], Y0[i])
            ec = d11 / d00
            tc = np.sqrt((d01 / d11) * (d00 / d10))
            rows.append({entity: e, "from": t0, "to": t1, "EC": ec, "TC": tc, "Malmquist": ec * tc})
    return pd.DataFrame(rows)


def cumulative(m: pd.DataFrame, entity="entity") -> pd.DataFrame:
    """Chain the period indices into cumulative productivity levels (first year = 1)."""
    return m.sort_values("to").groupby(entity)[["EC", "TC", "Malmquist"]].prod()
