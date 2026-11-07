"""Cut-off setting: PD-based versus quantile (rank-based) strategies.

* PD-based: reject when PD > threshold. Needs calibrated PDs, so it is reserved
  for models in classes A and B.
* Quantile: reject a fixed share of the riskiest applications. Uses only the
  ranking, so it is the fallback for class C models whose PDs cannot be trusted.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "break_even_pd",
    "cutoff_outcome",
    "pd_cutoff_table",
    "quantile_cutoff_table",
    "best_cutoff",
    "recommended_strategy",
]


def break_even_pd(margin: float, lgd: float) -> float:
    """PD at which a loan's expected profit is zero.

    Accepting a loan earns ``margin`` (per unit of exposure) if it performs and loses
    ``lgd`` if it defaults, so expected profit (1 - PD) * margin - PD * lgd is
    positive only below PD* = margin / (margin + lgd).
    """
    if margin <= 0 or lgd <= 0:
        raise ValueError("margin and lgd must be positive")
    return margin / (margin + lgd)


def cutoff_outcome(y_true, reject_mask, margin: float | None = None, lgd: float | None = None, ead=1.0) -> dict:
    """What a rejection rule does on historical outcomes."""
    y = np.asarray(y_true, dtype=float)
    rej = np.asarray(reject_mask, dtype=bool)
    acc = ~rej
    ead = np.broadcast_to(np.asarray(ead, dtype=float), y.shape)
    out = {
        "reject_share": float(rej.mean()),
        "defaults_avoided": float(y[rej].sum() / y.sum()) if y.sum() else np.nan,
        "bad_rate_rejected": float(y[rej].mean()) if rej.any() else np.nan,
        "bad_rate_accepted": float(y[acc].mean()) if acc.any() else np.nan,
    }
    if margin is not None and lgd is not None:
        profit = np.where(y[acc] == 1, -lgd, margin) * ead[acc]
        out["profit"] = float(profit.sum())
    return out


def pd_cutoff_table(y_true, pd, thresholds, **economics) -> list[dict]:
    pd = np.asarray(pd, dtype=float)
    return [{"pd_threshold": float(t), **cutoff_outcome(y_true, pd > t, **economics)} for t in thresholds]


def quantile_cutoff_table(y_true, pd, reject_shares, **economics) -> list[dict]:
    pd = np.asarray(pd, dtype=float)
    order = np.argsort(-pd, kind="mergesort")
    rows = []
    for s in reject_shares:
        mask = np.zeros(len(pd), dtype=bool)
        mask[order[: int(round(s * len(pd)))]] = True
        rows.append({"target_reject_share": float(s), **cutoff_outcome(y_true, mask, **economics)})
    return rows


def best_cutoff(table: list[dict], key: str = "profit") -> dict:
    """Row of a cut-off table with the highest ``key`` (historical back-test)."""
    return max(table, key=lambda r: r[key])


def recommended_strategy(model_class: str) -> str:
    return {
        "A": "pd",
        "B": "pd",  # with quarterly monitoring and back-testing of the threshold
        "C": "quantile",
        "D": "none",  # expert support only, no automated decisions
        "E": "none",
    }[model_class]
