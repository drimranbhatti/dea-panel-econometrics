"""effpanel - efficiency analysis and panel econometrics.

dea       : Data Envelopment Analysis (CCR/BCC, input/output orientation, scale efficiency, peers)
malmquist : Malmquist productivity index with efficiency-change / technical-change decomposition
panel     : pooled OLS, fixed / random effects, Hausman and Breusch-Pagan LM tests, fractional logit
"""
from . import dea, malmquist, panel

__all__ = ["dea", "malmquist", "panel"]
__version__ = "1.0.0"
