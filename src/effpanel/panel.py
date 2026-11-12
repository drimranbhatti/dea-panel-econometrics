"""Panel-data regressions for second-stage efficiency analysis (wraps linearmodels / statsmodels)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects, compare
from scipy import stats


def _xy(df, y, xs, const=True):
    X = df[xs]
    return df[y], (sm.add_constant(X) if const else X)


def fit_all(df: pd.DataFrame, y: str, xs: list[str], cluster=True) -> dict:
    """Pooled OLS, entity FE, two-way FE and RE. ``df`` must have a (entity, time) MultiIndex."""
    yv, X = _xy(df, y, xs)
    cov = dict(cov_type="clustered", cluster_entity=True) if cluster else dict(cov_type="robust")
    return {
        "Pooled OLS": PooledOLS(yv, X).fit(**cov),
        "Fixed effects (entity)": PanelOLS(yv, X, entity_effects=True, drop_absorbed=True).fit(**cov),
        "Fixed effects (two-way)": PanelOLS(yv, X, entity_effects=True, time_effects=True, drop_absorbed=True).fit(**cov),
        "Random effects": RandomEffects(yv, X).fit(**cov),
    }


def hausman(df: pd.DataFrame, y: str, xs: list[str]) -> dict:
    """Robust (Mundlak / Wooldridge) form of the Hausman test of fixed vs random effects.

    Augment the random-effects regression with entity means of the time-varying regressors
    (Mundlak, 1978) and Wald-test that their coefficients are jointly zero, using
    entity-clustered standard errors. H0: effects uncorrelated with regressors (RE consistent).
    Unlike the classical Hausman statistic this is always non-negative and robust to
    heteroskedasticity and serial correlation (Wooldridge, 2010, sec. 10.7.3).
    """
    d = df[[y] + xs].copy()
    means = d[xs].groupby(level=0).transform("mean").add_prefix("mean_")
    varying = [c for c in xs if (d[c] - means["mean_" + c]).abs().max() > 1e-12]
    aug = pd.concat([d, means[["mean_" + c for c in varying]]], axis=1)
    yv, X = _xy(aug, y, xs + ["mean_" + c for c in varying])
    res = RandomEffects(yv, X).fit(cov_type="clustered", cluster_entity=True)
    names = ["mean_" + c for c in varying]
    b = res.params[names].values
    V = res.cov.loc[names, names].values
    stat = float(b @ np.linalg.solve(V, b))
    dof = len(names)
    p = float(1 - stats.chi2.cdf(stat, dof))
    return {"chi2": stat, "dof": dof, "p_value": p, "preferred": "Fixed effects" if p < 0.05 else "Random effects"}


def breusch_pagan_lm(df: pd.DataFrame, y: str, xs: list[str]) -> dict:
    """Breusch-Pagan (1980) LM test for random effects vs pooled OLS (balanced-panel formula,
    applied to the entity-average structure). Rejection means individual effects matter."""
    yv, X = _xy(df, y, xs)
    e = PooledOLS(yv, X).fit().resids
    g = e.groupby(level=0)
    n, T = g.ngroups, g.size().mean()
    lm = n * T / (2 * (T - 1)) * ((g.sum() ** 2).sum() / (e**2).sum() - 1) ** 2
    return {"LM": float(lm), "p_value": float(1 - stats.chi2.cdf(lm, 1))}


def fractional_logit(df: pd.DataFrame, y: str, xs: list[str]):
    """Papke-Wooldridge (1996) fractional logit for scores in [0, 1], clustered by entity."""
    yv, X = _xy(df, y, xs)
    groups = pd.factorize(df.index.get_level_values(0))[0]
    return sm.GLM(yv.values, X.values, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})


def comparison_table(results: dict):
    return compare(results, stars=True, precision="std_errors")
