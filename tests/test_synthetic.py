import numpy as np
from effpanel import synthetic, panel


def test_fe_recovers_intangibles_effect():
    df = synthetic.simulate(n_firms=600, seed=1, dirty=False)
    d = df[df.year.between(2000, 2019)].set_index(["firm", "year"])
    xs = list(synthetic.TRUE)
    res = panel.fit_all(d, "d1tc", xs)
    fe = res["Fixed effects (entity)"].params["x1intan"]
    pooled = res["Pooled OLS"].params["x1intan"]
    assert abs(fe - synthetic.TRUE["x1intan"]) < 0.06
    assert abs(pooled - synthetic.TRUE["x1intan"]) > abs(fe - synthetic.TRUE["x1intan"])
