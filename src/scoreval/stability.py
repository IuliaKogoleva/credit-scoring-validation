"""Population Stability Index (PSI) for score distributions."""

from __future__ import annotations

import numpy as np

__all__ = ["psi", "psi_table"]


def psi_table(expected, actual, n_bins: int = 10, eps: float = 1e-4) -> list[dict]:
    """Per-bin PSI contributions.

    Bins are deciles of the ``expected`` (development) scores, so every bin starts
    with roughly 10% of the reference population.
    """
    expected = np.asarray(expected, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    exp_share = np.histogram(expected, edges)[0] / len(expected)
    act_share = np.histogram(actual, edges)[0] / len(actual)
    rows = []
    for i in range(len(edges) - 1):
        e, a = max(exp_share[i], eps), max(act_share[i], eps)
        rows.append(
            {
                "bin": i + 1,
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "expected_share": float(exp_share[i]),
                "actual_share": float(act_share[i]),
                "psi": float((a - e) * np.log(a / e)),
            }
        )
    return rows


def psi(expected, actual, n_bins: int = 10) -> float:
    """Population Stability Index between a reference and a current score sample.

    Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.
    """
    return float(sum(r["psi"] for r in psi_table(expected, actual, n_bins)))
