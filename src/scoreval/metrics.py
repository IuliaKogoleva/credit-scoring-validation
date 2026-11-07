"""Discrimination and calibration metrics for probability-of-default (PD) models.

Conventions used throughout the package:
    * ``y_true`` is 1 for a default ("bad") and 0 for a non-default ("good").
    * ``y_score`` / ``pd`` is the predicted probability of default; higher means riskier.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "bad_rate",
    "roc_auc",
    "gini",
    "ks_statistic",
    "pr_auc",
    "pr_auc_lift",
    "capture_rate",
    "brier_score",
    "naive_brier",
    "expected_calibration_error",
    "calibration_slope",
    "calibration_in_the_large",
    "hosmer_lemeshow",
    "binomial_test_by_grade",
    "bootstrap_ci",
]


def _as_arrays(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same length")
    if not np.isin(y_true, (0.0, 1.0)).all():
        raise ValueError("y_true must contain only 0 (good) and 1 (bad)")
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        raise ValueError("y_true must contain both classes")
    return y_true, y_score


def bad_rate(y_true) -> float:
    """Observed default rate, i.e. the PR-AUC baseline of a random model."""
    return float(np.mean(np.asarray(y_true, dtype=float)))


def roc_auc(y_true, y_score) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)
    return float(roc_auc_score(y_true, y_score))


def gini(y_true, y_score) -> float:
    """Gini coefficient (accuracy ratio) = 2 * ROC-AUC - 1."""
    return 2.0 * roc_auc(y_true, y_score) - 1.0


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov distance between score distributions of bads and goods."""
    y_true, y_score = _as_arrays(y_true, y_score)
    return float(stats.ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic)


def pr_auc(y_true, y_score) -> float:
    """Area under the precision-recall curve (average precision).

    A random model scores the bad rate, so the value must always be read against it.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    return float(average_precision_score(y_true, y_score))


def pr_auc_lift(y_true, y_score) -> float:
    """PR-AUC divided by its random baseline (the bad rate)."""
    return pr_auc(y_true, y_score) / bad_rate(y_true)


def capture_rate(y_true, y_score, top: float = 0.2) -> float:
    """Share of all defaults found in the riskiest ``top`` fraction of the population."""
    if not 0 < top <= 1:
        raise ValueError("top must be in (0, 1]")
    y_true, y_score = _as_arrays(y_true, y_score)
    n_top = int(np.ceil(top * len(y_score)))
    order = np.argsort(-y_score, kind="mergesort")
    return float(y_true[order[:n_top]].sum() / y_true.sum())


def brier_score(y_true, y_score) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)
    return float(np.mean((y_score - y_true) ** 2))


def naive_brier(y_true) -> float:
    """Brier score of the naive forecast that gives everyone the average bad rate."""
    p = bad_rate(y_true)
    return p * (1.0 - p)


def _bins(y_score, n_bins: int, strategy: str):
    if strategy == "quantile":
        edges = np.unique(np.quantile(y_score, np.linspace(0, 1, n_bins + 1)))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    return np.clip(np.searchsorted(edges, y_score, side="right") - 1, 0, len(edges) - 2)


def expected_calibration_error(y_true, y_score, n_bins: int = 10, strategy: str = "quantile") -> float:
    """Expected Calibration Error: population-weighted mean |observed - predicted| over bins.

    Quantile (equal-frequency) bins are the default because PDs of low-default
    portfolios crowd into the first few uniform bins.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    idx = _bins(y_score, n_bins, strategy)
    ece = 0.0
    for b in np.unique(idx):
        mask = idx == b
        ece += mask.mean() * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(ece)


def calibration_slope(y_true, y_score, eps: float = 1e-6) -> float:
    """Slope of a logistic recalibration of the outcome on logit(PD).

    1.0 is perfect; < 1 means PDs are too extreme (over-fitted), > 1 means too flat.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    p = np.clip(y_score, eps, 1 - eps)
    x = np.log(p / (1 - p))
    beta = np.array([0.0, 1.0])
    design = np.column_stack([np.ones_like(x), x])
    for _ in range(100):  # Newton-Raphson for a 2-parameter logistic regression
        mu = 1.0 / (1.0 + np.exp(-(design @ beta)))
        w = mu * (1 - mu)
        hess = design.T @ (design * w[:, None])
        grad = design.T @ (y_true - mu)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[1])


def calibration_in_the_large(y_true, y_score) -> dict:
    """Observed vs expected number of defaults over the whole sample.

    The calibration slope is blind to a uniform shift of all PDs, and an absolute
    ECE threshold is loose when the bad rate is low, so the overall level of the
    PDs is checked separately with a two-sided binomial test.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    observed, expected, n = int(y_true.sum()), float(y_score.sum()), len(y_true)
    p_value = stats.binomtest(observed, n, expected / n, alternative="two-sided").pvalue
    return {"observed": observed, "expected": expected, "oe_ratio": observed / expected, "p_value": float(p_value)}


def hosmer_lemeshow(y_true, y_score, n_bins: int = 10) -> dict:
    """Hosmer-Lemeshow goodness-of-fit test on equal-frequency bins."""
    y_true, y_score = _as_arrays(y_true, y_score)
    idx = _bins(y_score, n_bins, "quantile")
    stat = 0.0
    groups = np.unique(idx)
    for b in groups:
        mask = idx == b
        n, obs, exp = mask.sum(), y_true[mask].sum(), y_score[mask].sum()
        p_bar = exp / n
        if 0 < p_bar < 1:
            stat += (obs - exp) ** 2 / (n * p_bar * (1 - p_bar))
    dof = len(groups) - 2
    return {"statistic": float(stat), "dof": int(dof), "p_value": float(stats.chi2.sf(stat, dof))}


def binomial_test_by_grade(y_true, y_score, grades) -> list[dict]:
    """One-sided binomial test per rating grade: are defaults higher than the mean PD predicts?"""
    y_true, y_score = _as_arrays(y_true, y_score)
    grades = np.asarray(grades)
    rows = []
    for g in np.unique(grades):
        mask = grades == g
        n, k, pd_mean = int(mask.sum()), int(y_true[mask].sum()), float(y_score[mask].mean())
        rows.append(
            {
                "grade": g,
                "n": n,
                "defaults": k,
                "observed_dr": k / n,
                "mean_pd": pd_mean,
                "p_value": float(stats.binomtest(k, n, pd_mean, alternative="greater").pvalue),
            }
        )
    return rows


def bootstrap_ci(metric, y_true, y_score, n_boot: int = 500, alpha: float = 0.05, seed: int = 0, **kwargs):
    """Percentile bootstrap confidence interval for any ``metric(y_true, y_score)``.

    Resampling is stratified so each replicate keeps the observed number of defaults,
    which matters when there are only a few dozen of them.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    rng = np.random.default_rng(seed)
    bad_idx, good_idx = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    values = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate(
            [rng.choice(bad_idx, len(bad_idx), replace=True), rng.choice(good_idx, len(good_idx), replace=True)]
        )
        values[i] = metric(y_true[idx], y_score[idx], **kwargs)
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, y_score, **kwargs)), float(lo), float(hi)
