"""Synthetic consumer-loan data with a low default rate.

Real bank data cannot be published, so the examples run on a simulated portfolio
that reproduces the properties that matter for validation: ~2% defaults, a
non-linear default mechanism, noise, and an optional drift for monitoring samples.
Swap in any public dataset (e.g. Kaggle "Give Me Some Credit") with a 0/1 target.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq

__all__ = ["make_credit_portfolio", "FEATURES"]

FEATURES = [
    "age",
    "monthly_income",
    "debt_to_income",
    "credit_history_months",
    "delinquencies_2y",
    "revolving_utilization",
    "loan_amount",
    "employment_years",
    "inquiries_6m",
]


def _simulate(rng: np.random.Generator, n: int, drift: float):
    """Applicant features and the (intercept-free) log-odds of default."""
    age = np.clip(rng.normal(38 - 3 * drift, 11, n), 18, 75)
    income = np.exp(rng.normal(7.4 - 0.1 * drift, 0.55, n))
    dti = np.clip(rng.beta(2.2 + 1.5 * drift, 5.0, n), 0, 0.95)
    history = np.clip(rng.gamma(2.5, 30, n) * (age - 17) / 25 * (1 - 0.3 * drift), 0, 600)
    delinq = rng.poisson(0.25 + 0.25 * drift + 0.6 * dti, n)
    util = np.clip(rng.beta(1.6 + drift, 3.0, n) + 0.15 * (delinq > 0), 0, 1.3)
    loan = np.exp(rng.normal(8.3, 0.7, n))
    employment = np.clip(rng.gamma(2.0, 2.5, n) * (age - 17) / 20, 0, 45)
    inquiries = rng.poisson(0.8 + 1.2 * drift + 0.8 * util, n)

    signal = (
        -0.035 * (age - 38)
        - 0.70 * (np.log(income) - 7.4)
        + 3.2 * dti
        + 2.5 * np.maximum(dti - 0.5, 0)  # leverage hurts faster above 50%
        - 0.0025 * history
        + 0.55 * np.minimum(delinq, 4)
        + 1.6 * util
        + 0.20 * (np.log(loan) - 8.3)
        - 0.05 * np.minimum(employment, 15)
        + 0.18 * np.minimum(inquiries, 6)
    )
    logit = 1.15 * signal + rng.normal(0, 0.6, n)  # noise = unobserved heterogeneity
    features = pd.DataFrame(
        {
            "age": age.round(0),
            "monthly_income": income.round(0),
            "debt_to_income": dti.round(3),
            "credit_history_months": history.round(0),
            "delinquencies_2y": delinq,
            "revolving_utilization": util.round(3),
            "loan_amount": loan.round(0),
            "employment_years": employment.round(1),
            "inquiries_6m": inquiries,
        }
    )
    return features, logit


@lru_cache(maxsize=None)
def _intercept(target_bad_rate: float) -> float:
    """Intercept giving ``target_bad_rate`` on a large drift-free reference book."""
    _, ref = _simulate(np.random.default_rng(12345), 200_000, 0.0)
    return brentq(lambda c: np.mean(1 / (1 + np.exp(-(ref + c)))) - target_bad_rate, -20, 5)


def make_credit_portfolio(n: int = 50_000, target_bad_rate: float = 0.021, drift: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Simulate a loan book with a 0/1 ``default`` column.

    ``drift`` in [0, 1] worsens the applicant mix (higher leverage, more inquiries,
    thinner files) to mimic a later monitoring period. The intercept is fixed on the
    drift-free book, so a drifted book also defaults more often.
    """
    rng = np.random.default_rng(seed)
    features, logit = _simulate(rng, n, drift)
    p = 1 / (1 + np.exp(-(logit + _intercept(target_bad_rate))))
    features["default"] = rng.binomial(1, p)
    return features
