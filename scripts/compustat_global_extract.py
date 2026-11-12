"""Build the firm-level research panel from Compustat Global (requires a WRDS subscription).

Usage:  pip install wrds  &&  python scripts/compustat_global_extract.py --out data/private/oic_firms.csv
The output is NOT committed (data/private/ is git-ignored): Compustat's licence forbids redistribution.
Variable definitions match effpanel.synthetic and notebook 04:
  d1tc     trade credit supplied      = receivables (RECT) / sales (SALE)
  x1intan  intangibility              = intangible assets (INTANO) / total assets (AT)
  x2size   size                       = ln(total assets, USD)
  x3pay    trade credit received      = accounts payable (AP) / total assets
  x4inv    inventory intensity        = inventories (INVT) / total assets
  x5chf    cash holdings              = cash & short-term investments (CHE) / total assets
  x6profit profitability              = operating income (OIADP) / total assets
  x7growth sales growth               = SALE_t / SALE_t-1 - 1
  x8age    firm age                   = ln(1 + years since first appearance in Compustat)
  x9gdp    country income             = ln(GDP per capita), merged from the World Bank WDI
"""
import argparse
import numpy as np
import pandas as pd

OIC_ISO3 = pd.read_csv("data/oic_members.csv")["iso3"].tolist()

SQL = """
SELECT gvkey, fyear, loc AS iso3, curcd, gsector, at, sale, rect, intano, ap, invt, che, oiadp
FROM comp.g_funda
WHERE indfmt = 'INDL' AND datafmt = 'HIST_STD' AND consol = 'C' AND popsrc = 'I'
  AND fyear BETWEEN 1999 AND 2021 AND loc IN ({codes})
"""


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["gvkey", "fyear"]).drop_duplicates(["gvkey", "fyear"], keep="last")
    df = df[(df["at"] > 0) & (df["sale"] > 0)]
    out = pd.DataFrame({"country": df["iso3"], "year": df["fyear"], "firm": df["gvkey"], "sector": df["gsector"]})
    out["d1tc"] = df["rect"] / df["sale"]
    out["x1intan"] = df["intano"].fillna(0) / df["at"]
    out["x2size"] = np.log(df["at"])                      # convert to USD with comp.g_exrt_mth before this step if needed
    out["x3pay"], out["x4inv"], out["x5chf"] = df["ap"] / df["at"], df["invt"] / df["at"], df["che"] / df["at"]
    out["x6profit"] = df["oiadp"] / df["at"]
    out["x7growth"] = df.groupby("gvkey")["sale"].pct_change()
    out["x8age"] = np.log1p(df["fyear"] - df.groupby("gvkey")["fyear"].transform("min"))
    return out


if __name__ == "__main__":
    import wrds
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/private/oic_firms.csv"); a = ap.parse_args()
    db = wrds.Connection()
    raw = db.raw_sql(SQL.format(codes=",".join(f"'{c}'" for c in OIC_ISO3)))
    build(raw).to_csv(a.out, index=False)
    print("saved", a.out)
