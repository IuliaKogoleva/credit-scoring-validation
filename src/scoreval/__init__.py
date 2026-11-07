"""scoreval: validation toolkit for credit scoring (PD) models on imbalanced data."""

from .cutoff import break_even_pd, pd_cutoff_table, quantile_cutoff_table, recommended_strategy
from .metrics import (
    bad_rate,
    bootstrap_ci,
    brier_score,
    calibration_in_the_large,
    calibration_slope,
    capture_rate,
    expected_calibration_error,
    gini,
    hosmer_lemeshow,
    ks_statistic,
    pr_auc,
    pr_auc_lift,
    roc_auc,
)
from .risk_tiering import ModelMetrics, Thresholds, Verdict, classify, compute_metrics, validate
from .stability import psi, psi_table

__version__ = "0.1.0"

__all__ = [
    "bad_rate",
    "bootstrap_ci",
    "break_even_pd",
    "brier_score",
    "calibration_in_the_large",
    "calibration_slope",
    "capture_rate",
    "classify",
    "compute_metrics",
    "expected_calibration_error",
    "gini",
    "hosmer_lemeshow",
    "ks_statistic",
    "ModelMetrics",
    "pd_cutoff_table",
    "pr_auc",
    "pr_auc_lift",
    "psi",
    "psi_table",
    "quantile_cutoff_table",
    "recommended_strategy",
    "roc_auc",
    "Thresholds",
    "validate",
    "Verdict",
]
