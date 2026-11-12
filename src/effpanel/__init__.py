"""effpanel - efficiency analysis and panel econometrics.

dea       : Data Envelopment Analysis (CCR/BCC, input/output orientation, scale efficiency, peers)
malmquist : Malmquist productivity index with efficiency-change / technical-change decomposition
panel     : pooled OLS, fixed / random effects, Hausman and Breusch-Pagan LM tests, fractional logit
synthetic : simulated Compustat-style firm panel with known coefficients (licensed data cannot be shared)
"""
from . import dea, malmquist, panel, synthetic

__all__ = ["dea", "malmquist", "panel", "synthetic"]
__version__ = "1.0.0"
