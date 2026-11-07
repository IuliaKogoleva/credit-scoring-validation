"""Two-step model-risk classification of PD models.

Implements the methodology from Tkatchev & Kogoleva, "Beyond Standard Metrics:
Comprehensive Validation of Credit Scoring Models", Bankovsky Vestnik No. 8 (757), 2026:

    Step 1. Critical criteria (veto factors). Any breach rejects the model outright.
    Step 2. Non-critical criteria scored green / yellow / red (penalty levels).

The PR-AUC lift over its random baseline (the bad rate), the vetoes and the
penalty counts then map the model to one of five classes, A (full trust) to E (rejected).
All thresholds live in :class:`Thresholds` so a validation team can tune them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from . import metrics as m
from .stability import psi as _psi

__all__ = [
    "Thresholds",
    "ModelMetrics",
    "Finding",
    "Verdict",
    "lift_grade",
    "compute_metrics",
    "check_vetoes",
    "score_penalties",
    "classify",
    "validate",
]

GREEN, YELLOW, RED, INFO = "green", "yellow", "red", "info"


@dataclass(frozen=True)
class Thresholds:
    # Veto factors (Table 4)
    veto_lift: float = 2.0
    veto_slope_low: float = 0.3
    veto_slope_high: float = 2.0
    veto_psi: float = 0.50
    veto_capture_top10: float = 0.15
    # Penalty bands (Table 5)
    ece_green: float = 0.05
    ece_yellow: float = 0.10
    brier_rel_tol: float = 0.01  # |Brier / naive - 1| within this is "equal to the naive benchmark"
    capture20_green: float = 0.45
    capture20_yellow: float = 0.35
    psi_green: float = 0.10
    psi_yellow: float = 0.25
    slope_green: tuple = (0.8, 1.2)
    slope_yellow: tuple = (0.5, 1.5)
    abs_pr_auc: float = 0.5
    # The absolute "PR-AUC > 0.5" check (Table 5, last row) penalises almost every
    # low-default model, so it is reported but not counted unless switched on.
    count_abs_pr_auc: bool = False
    # Calibration-in-the-large (observed / expected defaults). Not part of the published
    # criteria, so reported only unless switched on; then a significant gap (p < oe_alpha)
    # larger than oe_tol is yellow and larger than 2 * oe_tol is red.
    count_oe: bool = False
    oe_tol: float = 0.20
    oe_alpha: float = 0.05
    # Lift scale (Table 2)
    lift_scale: tuple = ((2.0, "Very weak"), (3.0, "Weak"), (5.0, "Acceptable"), (10.0, "Good"))


DEFAULT_THRESHOLDS = Thresholds()


@dataclass
class ModelMetrics:
    bad_rate: float
    roc_auc: float
    pr_auc: float
    pr_auc_lift: float
    capture_top10: float
    capture_top20: float
    brier: float
    naive_brier: float
    ece: float
    calibration_slope: float
    psi: float | None = None
    oe_ratio: float | None = None
    oe_p_value: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Finding:
    criterion: str
    value: float | None
    level: str
    rule: str


@dataclass
class Verdict:
    model_class: str
    title: str
    allowed_use: str
    lift_grade: str
    metrics: ModelMetrics
    vetoes: list[Finding] = field(default_factory=list)
    penalties: list[Finding] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def yellow(self) -> int:
        return sum(p.level == YELLOW for p in self.penalties)

    @property
    def red(self) -> int:
        return sum(p.level == RED for p in self.penalties)

    def summary(self) -> str:
        mm = self.metrics
        lines = [
            f"Class {self.model_class}: {self.title}",
            f"  PR-AUC {mm.pr_auc:.3f} = {mm.pr_auc_lift:.2f}x baseline ({self.lift_grade}); ROC-AUC {mm.roc_auc:.3f}",
            f"  Vetoes: {len(self.vetoes) or 'none'}" + "".join(f"\n    - {v.criterion}: {v.rule}" for v in self.vetoes),
            f"  Penalties: {self.yellow} yellow, {self.red} red",
        ]
        lines += [f"    - {p.criterion} = {_fmt(p.value)} -> {p.level} ({p.rule})" for p in self.penalties if p.level in (YELLOW, RED)]
        if mm.oe_ratio is not None:
            lines.append(f"  Observed/expected defaults: {mm.oe_ratio:.2f} (p = {mm.oe_p_value:.3f})")
        lines += [f"  Why: {r}" for r in self.reasons]
        lines.append(f"  Allowed use: {self.allowed_use}")
        return "\n".join(lines)


CLASS_INFO = {
    "A": ("Approved, full trust", "Automatic PD-based cut-off; suitable for IRB use and economic capital."),
    "B": (
        "Approved with limitations",
        "PD-based cut-off allowed with quarterly monitoring; cut-off may be adjusted to observed defaults.",
    ),
    "C": (
        "Conditionally approved, recalibration required",
        "Quantile cut-off only (reject a fixed share of the riskiest applications); PDs not used for decisions; "
        "recalibration plan required.",
    ),
    "D": (
        "Weak model, temporary use",
        "Auxiliary indicator for expert judgement only, no automated decisions; limited lifetime (e.g. 6 months).",
    ),
    "E": ("Rejected", "Not allowed in production; return for redevelopment or replace."),
}


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def lift_grade(lift: float, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> str:
    for upper, label in thresholds.lift_scale:
        if lift < upper:
            return label
    return "Excellent"


def compute_metrics(y_true, pd, reference_pd=None, n_bins: int = 10) -> ModelMetrics:
    """All inputs needed by the classification, from outcomes and predicted PDs.

    ``reference_pd`` are scores on the development sample; when given, PSI is
    computed against them, otherwise the stability criteria are skipped.
    """
    y_true = np.asarray(y_true, dtype=float)
    pd = np.asarray(pd, dtype=float)
    citl = m.calibration_in_the_large(y_true, pd)
    return ModelMetrics(
        bad_rate=m.bad_rate(y_true),
        roc_auc=m.roc_auc(y_true, pd),
        pr_auc=m.pr_auc(y_true, pd),
        pr_auc_lift=m.pr_auc_lift(y_true, pd),
        capture_top10=m.capture_rate(y_true, pd, 0.10),
        capture_top20=m.capture_rate(y_true, pd, 0.20),
        brier=m.brier_score(y_true, pd),
        naive_brier=m.naive_brier(y_true),
        ece=m.expected_calibration_error(y_true, pd, n_bins=n_bins),
        calibration_slope=m.calibration_slope(y_true, pd),
        psi=None if reference_pd is None else _psi(reference_pd, pd, n_bins=n_bins),
        oe_ratio=citl["oe_ratio"],
        oe_p_value=citl["p_value"],
    )


def check_vetoes(mm: ModelMetrics, t: Thresholds = DEFAULT_THRESHOLDS) -> list[Finding]:
    """Step 1: critical criteria. Any returned finding rejects the model."""
    out = []
    if mm.pr_auc_lift < t.veto_lift:
        out.append(Finding("PR-AUC lift", mm.pr_auc_lift, RED, f"< {t.veto_lift}x baseline: no practical predictive value"))
    if not t.veto_slope_low <= mm.calibration_slope <= t.veto_slope_high:
        out.append(
            Finding(
                "Calibration slope",
                mm.calibration_slope,
                RED,
                f"outside [{t.veto_slope_low}, {t.veto_slope_high}]: PDs unusable for segmentation or cut-offs",
            )
        )
    if mm.psi is not None and mm.psi > t.veto_psi:
        out.append(Finding("PSI", mm.psi, RED, f"> {t.veto_psi}: score distribution has broken down"))
    if mm.capture_top10 < t.veto_capture_top10:
        out.append(
            Finding("Capture rate, top 10%", mm.capture_top10, RED, f"< {t.veto_capture_top10:.0%}: no ranking where it matters most")
        )
    return out


def _band(value, green_ok, yellow_ok):
    return GREEN if green_ok(value) else YELLOW if yellow_ok(value) else RED


def score_penalties(mm: ModelMetrics, t: Thresholds = DEFAULT_THRESHOLDS) -> list[Finding]:
    """Step 2: non-critical criteria, each scored green / yellow / red."""
    out = [
        Finding(
            "ECE",
            mm.ece,
            _band(mm.ece, lambda v: v < t.ece_green, lambda v: v <= t.ece_yellow),
            f"green < {t.ece_green}, yellow {t.ece_green}-{t.ece_yellow}, red > {t.ece_yellow}",
        )
    ]
    ratio = mm.brier / mm.naive_brier
    out.append(
        Finding(
            "Brier score vs naive",
            mm.brier,
            GREEN if ratio < 1 - t.brier_rel_tol else YELLOW if ratio <= 1 + t.brier_rel_tol else RED,
            f"naive benchmark = {mm.naive_brier:.4f}; equal means within ±{t.brier_rel_tol:.0%}",
        )
    )
    out.append(
        Finding(
            "Capture rate, top 20%",
            mm.capture_top20,
            _band(mm.capture_top20, lambda v: v > t.capture20_green, lambda v: v >= t.capture20_yellow),
            f"green > {t.capture20_green:.0%}, yellow {t.capture20_yellow:.0%}-{t.capture20_green:.0%}, red < {t.capture20_yellow:.0%}",
        )
    )
    if mm.psi is None:
        out.append(Finding("PSI", None, INFO, "not computed: no reference (development) scores supplied"))
    else:
        out.append(
            Finding(
                "PSI",
                mm.psi,
                _band(mm.psi, lambda v: v < t.psi_green, lambda v: v <= t.psi_yellow),
                f"green < {t.psi_green}, yellow {t.psi_green}-{t.psi_yellow}, red {t.psi_yellow}-{t.veto_psi}",
            )
        )
    g_lo, g_hi = t.slope_green
    y_lo, y_hi = t.slope_yellow
    out.append(
        Finding(
            "Calibration slope",
            mm.calibration_slope,
            _band(mm.calibration_slope, lambda v: g_lo <= v <= g_hi, lambda v: y_lo <= v <= y_hi),
            f"green {g_lo}-{g_hi}, yellow {y_lo}-{g_lo} or {g_hi}-{y_hi}, red outside (but within veto bounds)",
        )
    )
    if mm.oe_ratio is not None:
        gap = abs(mm.oe_ratio - 1)
        significant = mm.oe_p_value is None or mm.oe_p_value < t.oe_alpha
        level = GREEN if not significant or gap <= t.oe_tol else YELLOW if gap <= 2 * t.oe_tol else RED
        out.append(
            Finding(
                "Observed / expected defaults",
                mm.oe_ratio,
                level if t.count_oe else INFO,
                f"p = {mm.oe_p_value:.3f}; flagged if significant and off by > {t.oe_tol:.0%}; "
                + ("counted" if t.count_oe else "reported only, not counted (see Thresholds)"),
            )
        )
    abs_level = GREEN if mm.pr_auc > t.abs_pr_auc else RED
    out.append(
        Finding(
            "Absolute PR-AUC",
            mm.pr_auc,
            abs_level if t.count_abs_pr_auc else INFO,
            f"green > {t.abs_pr_auc}; " + ("counted" if t.count_abs_pr_auc else "reported only, not counted (see Thresholds)"),
        )
    )
    return out


def classify(mm: ModelMetrics, t: Thresholds = DEFAULT_THRESHOLDS) -> Verdict:
    """Map metrics to class A-E (Table 6). Classes are tried from best to worst."""
    vetoes = check_vetoes(mm, t)
    penalties = score_penalties(mm, t)
    yellow = sum(p.level == YELLOW for p in penalties)
    red = sum(p.level == RED for p in penalties)
    lift, slope, ece, cr20 = mm.pr_auc_lift, mm.calibration_slope, mm.ece, mm.capture_top20
    g_lo, g_hi = t.slope_green
    y_lo, y_hi = t.slope_yellow

    def verdict(cls, reasons):
        title, use = CLASS_INFO[cls]
        return Verdict(cls, title, use, lift_grade(lift, t), mm, vetoes, penalties, reasons)

    if vetoes:
        return verdict("E", [f"veto: {v.criterion}" for v in vetoes])
    if red >= 3:
        return verdict("E", [f"{red} red penalties (3 or more)"])

    a_checks = {
        "PR-AUC >= 5x baseline": lift >= 5,
        f"slope in {g_lo}-{g_hi}": g_lo <= slope <= g_hi,
        f"ECE < {t.ece_green}": ece < t.ece_green,
        f"capture top 20% > {t.capture20_green:.0%}": cr20 > t.capture20_green,
        "at most 1 yellow, no red": yellow <= 1 and red == 0,
    }
    if all(a_checks.values()):
        return verdict("A", ["all class A conditions met"])

    b_checks = {
        "PR-AUC >= 5x baseline": lift >= 5,
        f"slope in {y_lo}-{y_hi}": y_lo <= slope <= y_hi,
        f"ECE < {t.ece_yellow}": ece < t.ece_yellow,
        f"capture top 20% > {t.capture20_yellow:.0%}": cr20 > t.capture20_yellow,
        "at most 2 yellow, no red": yellow <= 2 and red == 0,
    }
    if all(b_checks.values()):
        return verdict("B", ["not class A: " + ", ".join(k for k, ok in a_checks.items() if not ok)])

    failed_b = ", ".join(k for k, ok in b_checks.items() if not ok)
    if lift >= 3 and red <= 1:
        return verdict("C", [f"not class B: {failed_b}"])
    if lift >= 2 and red <= 2:
        return verdict("D", [f"not class C: PR-AUC lift {lift:.2f}x < 3x" if lift < 3 else f"not class C: {red} red penalties"])
    return verdict("E", ["conditions for class D not met"])


def validate(y_true, pd, reference_pd=None, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> Verdict:
    """One call: compute metrics on a validation sample and classify the model."""
    return classify(compute_metrics(y_true, pd, reference_pd), thresholds)
