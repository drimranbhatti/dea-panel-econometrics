import numpy as np
import pandas as pd
import pytest

from effpanel import dea, malmquist, panel


def test_single_input_output_crs_is_productivity_ratio():
    X = np.array([[2.0], [4.0], [5.0], [8.0]]); Y = np.array([[2.0], [3.0], [5.0], [4.0]])
    ratio = (Y / X).ravel(); expected = ratio / ratio.max()
    assert np.allclose(dea.efficiency(X, Y)["score"], expected)
    assert np.allclose(dea.efficiency(X, Y, orientation="output")["score"], expected)   # CRS: same in both orientations


def test_vrs_at_least_crs_and_extremes_efficient():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 10, (20, 2)); Y = rng.uniform(1, 10, (20, 1))
    s = dea.scale_efficiency(X, Y)
    assert (s["VRS (pure technical)"] >= s["CRS (technical)"] - 1e-9).all()
    assert ((s["CRS (technical)"] > 0) & (s["CRS (technical)"] <= 1 + 1e-9)).all()
    assert s["VRS (pure technical)"].max() == pytest.approx(1.0)


def test_textbook_example_cooper_seiford_tone():
    # Cooper, Seiford & Tone (2007), Table 3.2-style: 2 inputs, 1 unit output
    X = np.array([[4, 3], [7, 3], [8, 1], [4, 2], [2, 4], [10, 1]], float); Y = np.ones((6, 1))
    sc = dea.efficiency(X, Y, names=list("ABCDEF"))["score"]
    assert sc[["C", "D", "E"]].tolist() == pytest.approx([1, 1, 1])
    assert sc["A"] == pytest.approx(6 / 7, abs=1e-6)
    assert sc["B"] == pytest.approx(0.6316, abs=1e-4)


def test_malmquist_identity_and_tech_progress():
    base = pd.DataFrame({"entity": list("abc") * 2, "year": [1] * 3 + [2] * 3,
                         "x": [2.0, 4.0, 5.0] * 2, "y": [2.0, 3.0, 5.0] * 2})
    m = malmquist.malmquist(base, ["x"], ["y"])
    assert np.allclose(m["Malmquist"], 1)
    grown = base.copy(); grown.loc[grown.year == 2, "y"] *= 1.1     # everyone 10% more output
    m2 = malmquist.malmquist(grown, ["x"], ["y"])
    assert np.allclose(m2["TC"], 1.1) and np.allclose(m2["EC"], 1)


def test_hausman_prefers_fe_when_effects_correlated():
    rng = np.random.default_rng(1)
    n, T = 60, 10
    alpha = rng.normal(size=n)
    x = alpha[:, None] + rng.normal(size=(n, T))                      # regressor correlated with effect
    y = 2 * x + alpha[:, None] * 3 + rng.normal(size=(n, T))
    df = pd.DataFrame({"y": y.ravel(), "x": x.ravel()},
                      index=pd.MultiIndex.from_product([range(n), range(T)], names=["e", "t"]))
    assert panel.hausman(df, "y", ["x"])["preferred"] == "Fixed effects"
    fe = panel.fit_all(df, "y", ["x"])["Fixed effects (entity)"]
    assert fe.params["x"] == pytest.approx(2, abs=0.1)


def test_hausman_size_when_re_is_valid():
    rejections = 0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        n, T = 60, 8
        alpha = rng.normal(size=n)
        x = rng.normal(size=(n, T))                                     # independent of the effect
        y = 1.5 * x + alpha[:, None] + rng.normal(size=(n, T))
        df = pd.DataFrame({"y": y.ravel(), "x": x.ravel()},
                          index=pd.MultiIndex.from_product([range(n), range(T)], names=["e", "t"]))
        rejections += panel.hausman(df, "y", ["x"])["p_value"] < 0.05
    assert rejections <= 4                                              # ~5% nominal size
