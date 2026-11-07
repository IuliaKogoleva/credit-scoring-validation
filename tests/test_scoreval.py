import numpy as np
import pytest

from scoreval import (
    Thresholds,
    break_even_pd,
    calibration_slope,
    capture_rate,
    classify,
    expected_calibration_error,
    pr_auc,
    pr_auc_lift,
    psi,
    quantile_cutoff_table,
    roc_auc,
    validate,
)
from scoreval.data import make_credit_portfolio
from scoreval.risk_tiering import ModelMetrics


def metrics(**kw):
    base = dict(
        bad_rate=0.02,
        roc_auc=0.85,
        pr_auc=0.18,
        pr_auc_lift=9.0,
        capture_top10=0.35,
        capture_top20=0.52,
        brier=0.0190,
        naive_brier=0.0206,
        ece=0.02,
        calibration_slope=1.05,
        psi=0.06,
    )
    base.update(kw)
    return ModelMetrics(**base)


# --- Worked examples from the article -------------------------------------------------


def test_example_1_mortgage_is_class_a():
    assert classify(metrics()).model_class == "A"


def test_example_2_cards_is_class_c():
    v = classify(metrics(bad_rate=0.05, pr_auc=0.18, pr_auc_lift=3.6, calibration_slope=0.65, ece=0.08, capture_top20=0.38, psi=0.12))
    assert v.model_class == "C"
    assert v.red == 0 and v.yellow >= 3


def test_example_3_sme_is_rejected_by_three_vetoes():
    v = classify(metrics(bad_rate=0.04, pr_auc=0.06, pr_auc_lift=1.5, calibration_slope=2.3, psi=0.55))
    assert v.model_class == "E"
    assert {f.criterion for f in v.vetoes} == {"PR-AUC lift", "Calibration slope", "PSI"}


def test_approbation_sample_is_class_a():
    v = classify(metrics(bad_rate=0.021, pr_auc=0.11, pr_auc_lift=0.11 / 0.021, calibration_slope=1.08, ece=0.035, capture_top10=0.41, capture_top20=0.48, psi=0.07))
    assert v.model_class == "A" and v.yellow == 0 and v.red == 0


# --- Class boundaries -----------------------------------------------------------------


def test_class_b_when_slope_only_yellow():
    assert classify(metrics(calibration_slope=1.3)).model_class == "B"


def test_class_d_for_lift_between_2_and_3():
    assert classify(metrics(pr_auc_lift=2.5)).model_class == "D"


def test_three_reds_reject():
    v = classify(metrics(pr_auc_lift=4, ece=0.2, capture_top20=0.2, brier=0.03))
    assert v.red == 3 and v.model_class == "E"


def test_low_capture_in_top_decile_is_a_veto():
    assert classify(metrics(capture_top10=0.10)).model_class == "E"


def test_absolute_pr_auc_only_counts_when_enabled():
    mm = metrics(pr_auc=0.11)
    assert classify(mm).model_class == "A"
    assert classify(mm, Thresholds(count_abs_pr_auc=True)).red == 1


def test_missing_psi_is_reported_not_penalised():
    v = classify(metrics(psi=None))
    assert v.model_class == "A"
    assert any(p.criterion == "PSI" and p.level == "info" for p in v.penalties)


# --- Metrics --------------------------------------------------------------------------


def test_roc_auc_flatters_a_model_that_misses_defaults():
    """The article's illustration: 1,000 borrowers, 10 defaults, a model that ranks well overall."""
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(10), np.zeros(990)]
    score = np.r_[rng.normal(1.3, 1, 10), rng.normal(0, 1, 990)]
    assert roc_auc(y, score) > 0.75
    assert pr_auc(y, score) < 0.3
    assert pr_auc_lift(y, score) == pytest.approx(pr_auc(y, score) / 0.01)


def test_capture_rate():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert capture_rate(y, s, 0.2) == 0.5


def test_calibration_on_well_calibrated_scores():
    rng = np.random.default_rng(1)
    p = rng.beta(1, 40, 200_000)
    y = rng.binomial(1, p)
    assert calibration_slope(y, p) == pytest.approx(1.0, abs=0.05)
    assert expected_calibration_error(y, p) < 0.005


def test_slope_detects_overconfident_scores():
    rng = np.random.default_rng(2)
    p = rng.beta(1, 40, 200_000)
    y = rng.binomial(1, p)
    logit = np.log(p / (1 - p))
    overconfident = 1 / (1 + np.exp(-(2 * (logit - np.mean(logit)) + np.mean(logit))))
    assert calibration_slope(y, overconfident) == pytest.approx(0.5, abs=0.05)


def test_psi_is_zero_on_identical_and_large_on_shifted():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 50_000)
    assert psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert psi(a, a + 1.0) > 0.5


def test_break_even_pd():
    assert break_even_pd(margin=0.08, lgd=0.6) == pytest.approx(0.08 / 0.68)


def test_quantile_cutoff_rejects_requested_share():
    y = np.array([0, 1] * 50)
    s = np.linspace(0, 1, 100)
    row = quantile_cutoff_table(y, s, [0.1])[0]
    assert row["reject_share"] == pytest.approx(0.1)


def test_end_to_end_on_synthetic_data():
    from sklearn.linear_model import LogisticRegression

    from scoreval.data import FEATURES

    dev = make_credit_portfolio(30_000, seed=1)
    test = make_credit_portfolio(20_000, seed=2)
    model = LogisticRegression(max_iter=2000).fit(np.log1p(dev[FEATURES]), dev["default"])
    pd_dev = model.predict_proba(np.log1p(dev[FEATURES]))[:, 1]
    pd_test = model.predict_proba(np.log1p(test[FEATURES]))[:, 1]
    v = validate(test["default"], pd_test, reference_pd=pd_dev)
    assert v.model_class in {"A", "B", "C", "D", "E"}
    assert 0.01 < v.metrics.bad_rate < 0.04


def test_oe_ratio_flags_uniformly_understated_pds_only_when_enabled():
    rng = np.random.default_rng(5)
    p = rng.beta(1, 40, 100_000)
    y = rng.binomial(1, p)
    halved = p / 2
    mm = sv_compute(y, halved)
    assert mm.oe_ratio == pytest.approx(2.0, rel=0.05)
    # The published criteria alone do not penalise a uniform shift; the optional check does.
    assert classify(mm, Thresholds(count_oe=True)).red == classify(mm).red + 1


def sv_compute(y, p):
    from scoreval import compute_metrics

    return compute_metrics(y, p, reference_pd=p)
