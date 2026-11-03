"""pau-r: leverage-aware scale (cfg["pr_lev"] = c). The t-scales are fitted on inner-LOYO residuals, so they hold
the year-regression parameter error of a typical inner fold. A test year whose generic ballot x midterm lies far
from the training years (2018 trained on 2010-16; 2026 at GB 8.6) has a larger one. Adjust per predicted year:
  sig^2 <- sig^2 + c * s2 * (lev_T - mean_t lev_t) / (nu / (nu - 2)),  floored at 0.5 sig^2,
s2 = dof-corrected residual variance of the year regression, lev_T = x0'(A'A)^-1 x0, inner lev_t = h_tt/(1-h_tt)."""
import numpy as np
import model as M

_orig_predict = M.WinModel.predict


def _adj(self, df):
    st = getattr(self, "stage1_", None)
    if st is None or not hasattr(st, "yl_") or not hasattr(st, "_pr_train"):
        return None
    tr = st._pr_train
    yr = tr.groupby("year").first().loc[st.years_]
    x = (yr.gb * yr.midterm).values.astype(float)
    A = np.column_stack([np.ones(len(x)), x])
    G = np.linalg.inv(A.T @ A + np.diag([0, 1e-3]))
    b = G @ A.T @ st.fe_.values
    r = st.fe_.values - A @ b
    if len(x) <= 2:
        return None
    s2 = (r ** 2).sum() / (len(x) - 2)
    H = np.einsum("ij,jk,ik->i", A, G, A)
    inner = np.mean(H / np.clip(1 - H, 0.05, None))
    T = df.groupby("year").first()
    x0 = np.array([1.0, float(T.gb.iloc[0] * T.midterm.iloc[0])])
    lev = float(x0 @ G @ x0)
    nu = self.cfg.get("nu") or 10
    return self.cfg["pr_lev"] * s2 * (lev - inner) / (nu / (nu - 2))


def predict(self, df):
    if not self.cfg.get("pr_lev") or df.year.nunique() != 1 or not hasattr(self, "sig_"):
        return _orig_predict(self, df)
    add = _adj(self, df)
    if add is None:
        return _orig_predict(self, df)
    keep = dict(self.sig_)
    self.sig_ = {k: float(np.sqrt(max(v * v + add, 0.5 * v * v))) for k, v in keep.items()}
    try:
        return _orig_predict(self, df)
    finally:
        self.sig_ = keep


M.WinModel.predict = predict
