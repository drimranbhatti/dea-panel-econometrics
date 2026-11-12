"""Simulated firm-level panel with the structure of a Compustat Global sample of OIC firms.

Licensed Compustat data cannot be redistributed, so this generator produces a panel with the same
*shape* (country, sector, firm, year; trade credit, intangibles and controls), realistic data problems
(duplicate firm-years, extreme ratios, missing values) and **known** coefficients, so that the
estimation pipeline can be shown and its ability to recover the truth tested. The variable names
follow the research dataset; the values are artificial."""
from __future__ import annotations
import numpy as np
import pandas as pd

COUNTRIES = {"Malaysia": .30, "Indonesia": .13, "Pakistan": .11, "Turkey": .09, "Bangladesh": .05, "Saudi Arabia": .04,
             "Jordan": .04, "Egypt": .04, "Nigeria": .03, "Kuwait": .03, "Oman": .025, "Morocco": .02, "United Arab Emirates": .02,
             "Tunisia": .015, "Qatar": .01, "Bahrain": .01, "Kazakhstan": .01, "Cote d'Ivoire": .008, "Lebanon": .007,
             "Palestine": .006, "Senegal": .005, "Uganda": .005, "Gabon": .003, "Iraq": .003}
SECTORS = ["Industrials", "Consumer Discretionary", "Materials", "Consumer Staples", "Information Technology", "Health Care", "Energy"]
TRUE = {"x1intan": 0.25, "x2size": 0.015, "x3pay": 0.30, "x4inv": -0.10, "x5chf": -0.12, "x6profit": 0.08, "x7growth": 0.04, "x8age": 0.02}


def simulate(n_firms: int = 3600, years=(1999, 2021), seed: int = 0, dirty: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    countries = rng.choice(list(COUNTRIES), n_firms, p=np.array(list(COUNTRIES.values())) / sum(COUNTRIES.values()))
    sectors = rng.choice(SECTORS, n_firms, p=[.22, .2, .19, .15, .08, .08, .08])
    country_eff = {c: rng.normal(0, 0.04) for c in COUNTRIES}
    sector_eff = {s: e for s, e in zip(SECTORS, [0.05, 0.02, 0.03, -0.02, 0.06, 0.04, -0.03])}
    firm_eff = rng.normal(0, 0.06, n_firms)
    # intangibles correlated with the firm effect -> pooled OLS / RE biased, FE consistent
    intan_mean = np.clip(0.03 + 0.15 * firm_eff + rng.normal(0, 0.02, n_firms), 0, None)
    gdp_c = {c: rng.uniform(7.5, 11.0) for c in COUNTRIES}
    rows = []
    for i in range(n_firms):
        start = rng.integers(years[0], years[1] - 4); end = min(years[1], start + rng.integers(5, 23))
        age0 = rng.uniform(1.0, 3.0); size0 = rng.normal(6.5, 1.8)
        for t in range(start, end + 1):
            x = {"x1intan": np.clip(intan_mean[i] + rng.normal(0, 0.015), 0, 0.99), "x2size": size0 + 0.04 * (t - start) + rng.normal(0, 0.1),
                 "x3pay": np.clip(rng.normal(0.14, 0.06), 0, None), "x4inv": np.clip(rng.normal(0.20, 0.09), 0, None),
                 "x5chf": np.clip(rng.normal(0.08, 0.06), 0, 1), "x6profit": rng.normal(0.05, 0.07), "x7growth": rng.normal(0.08, 0.25),
                 "x8age": np.log1p(np.exp(age0) + (t - start))}
            y = 0.10 + sum(TRUE[k] * x[k] for k in TRUE) + firm_eff[i] + country_eff[countries[i]] + sector_eff[sectors[i]] \
                + 0.01 * np.sin(t / 3) + rng.normal(0, 0.05)
            rows.append({"country": countries[i], "year": t, "firm": f"F{i:05d}", "sector": sectors[i], "d1tc": max(y, 0.0), **x,
                         "x9gdp": gdp_c[countries[i]] + 0.02 * (t - years[0]) + rng.normal(0, 0.01)})
    df = pd.DataFrame(rows)
    if dirty:
        dup = df.sample(frac=0.01, random_state=seed)                          # duplicate firm-years
        df = pd.concat([df, dup], ignore_index=True)
        k = rng.choice(len(df), int(0.004 * len(df)), replace=False)          # ratio explosions (tiny denominators)
        df.loc[k, "x7growth"] = rng.lognormal(6, 2, len(k))
        k = rng.choice(len(df), int(0.003 * len(df)), replace=False)
        df.loc[k, "x3pay"] = rng.lognormal(4, 2, len(k))
        k = rng.choice(len(df), int(0.02 * len(df)), replace=False)
        df.loc[k, "x6profit"] = np.nan
    return df.sort_values(["firm", "year"]).reset_index(drop=True)
