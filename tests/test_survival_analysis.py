"""Tests for survival_analysis module."""

import numpy as np
import pandas as pd
import pytest

from hla_analysis.config import AnalysisConfig
from hla_analysis.survival_analysis import (
    SurvivalAnalyzer, fast_cox_single, _fit_survival_single,
)


class TestFastCoxSingle:
    """Tests for the custom Newton-Raphson Cox PH solver."""

    def test_basic_fit(self):
        """Test basic Cox PH fit returns expected keys."""
        rng = np.random.RandomState(42)
        n = 100
        time = rng.exponential(500, n)
        event = rng.choice([0, 1], n, p=[0.3, 0.7]).astype(float)
        X = rng.normal(0, 1, (n, 2))

        result = fast_cox_single(time, event, X)

        assert "beta" in result
        assert "se" in result
        assert "hr" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "pvalue" in result
        assert "concordance" in result
        assert "converged" in result

    def test_convergence(self):
        """Test that the solver converges for well-conditioned data."""
        rng = np.random.RandomState(42)
        n = 200
        X = rng.normal(0, 1, (n, 1))
        # Generate survival times with known effect
        true_beta = 0.5
        hazard = np.exp(X[:, 0] * true_beta) * 0.001
        time = rng.exponential(1.0 / hazard)
        event = np.ones(n)

        result = fast_cox_single(time, event, X, max_iter=100)
        assert result["converged"] is True

    def test_hr_is_exp_beta(self):
        """Test that HR = exp(beta)."""
        rng = np.random.RandomState(42)
        n = 200
        time = rng.exponential(500, n)
        event = rng.choice([0, 1], n, p=[0.2, 0.8]).astype(float)
        X = rng.normal(0, 1, (n, 1))

        result = fast_cox_single(time, event, X)
        if not np.isnan(result["beta"]):
            assert result["hr"] == pytest.approx(np.exp(result["beta"]), rel=1e-3)

    def test_compare_with_lifelines(self):
        """Test that custom solver gives similar results to lifelines."""
        from lifelines import CoxPHFitter

        rng = np.random.RandomState(42)
        n = 200
        time = rng.exponential(500, n)
        event = rng.choice([0, 1], n, p=[0.3, 0.7]).astype(float)
        x = rng.normal(0, 1, n)

        # Custom solver
        custom = fast_cox_single(time, event, x.reshape(-1, 1))

        # Lifelines
        df = pd.DataFrame({"T": time, "E": event, "x": x})
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(df, "T", "E")
        ll_beta = cph.summary.loc["x", "coef"]
        ll_se = cph.summary.loc["x", "se(coef)"]

        # Should be reasonably close (not exact due to implementation differences)
        if custom["converged"]:
            assert custom["beta"] == pytest.approx(ll_beta, abs=0.1)
            assert custom["se"] == pytest.approx(ll_se, abs=0.05)

    def test_handles_ties(self):
        """Test handling of tied event times."""
        rng = np.random.RandomState(42)
        n = 50
        # Create ties
        time = np.array([100]*10 + [200]*10 + [300]*10 + list(rng.exponential(500, 20)))
        event = rng.choice([0, 1], n, p=[0.3, 0.7]).astype(float)
        X = rng.normal(0, 1, (n, 1))

        result = fast_cox_single(time, event, X)
        assert "beta" in result  # Should not crash

    def test_single_feature(self):
        """Test with just one feature column."""
        rng = np.random.RandomState(42)
        n = 100
        time = rng.exponential(500, n)
        event = rng.choice([0, 1], n, p=[0.3, 0.7]).astype(float)
        X = rng.choice([0, 1, 2], (n, 1)).astype(float)

        result = fast_cox_single(time, event, X)
        assert result is not None


class TestFitSurvivalSingle:
    """Tests for the single-feature survival model fitter."""

    def test_basic_fit(self):
        """Test basic single-feature survival fit."""
        rng = np.random.RandomState(42)
        n = 100
        time = rng.exponential(500, n)
        event = rng.choice([0, 1], n, p=[0.3, 0.7]).astype(float)
        dosage = rng.choice([0, 1, 2], n, p=[0.5, 0.35, 0.15]).astype(float)

        result = _fit_survival_single(
            dosage, time, event, None, "test_feature",
            min_events=3, use_custom=True,
        )
        assert result is not None
        assert "feature" in result
        assert result["feature"] == "test_feature"

    def test_skip_low_events(self):
        """Test skipping features with too few carrier events."""
        n = 50
        time = np.random.exponential(500, n)
        event = np.zeros(n)
        event[:3] = 1  # Only 3 events
        dosage = np.zeros(n)
        dosage[0] = 1  # Only 1 carrier among events

        result = _fit_survival_single(dosage, time, event, None, "rare", min_events=5)
        assert result is None

    def test_skip_zero_variance(self):
        """Test skipping zero-variance features."""
        n = 50
        time = np.random.exponential(500, n)
        event = np.ones(n)
        dosage = np.ones(n)

        result = _fit_survival_single(dosage, time, event, None, "constant", min_events=1)
        assert result is None

    def test_lifelines_fallback(self):
        """Test lifelines solver path."""
        rng = np.random.RandomState(42)
        n = 100
        time = rng.exponential(500, n)
        event = rng.choice([0, 1], n, p=[0.3, 0.7]).astype(float)
        dosage = rng.choice([0, 1, 2], n, p=[0.5, 0.35, 0.15]).astype(float)

        result = _fit_survival_single(
            dosage, time, event, None, "test",
            min_events=3, use_custom=False, penalizer=0.01,
        )
        assert result is not None


class TestSurvivalAnalyzer:
    """Tests for the SurvivalAnalyzer class."""

    def test_analyze_stratum(self):
        """Test survival analysis on a stratum."""
        rng = np.random.RandomState(42)
        n = 150
        n_features = 8
        time = rng.exponential(500, n)
        event = rng.choice([0, 1], n, p=[0.3, 0.7]).astype(float)
        dosage = rng.choice([0, 1, 2], (n, n_features), p=[0.4, 0.4, 0.2]).astype(np.float32)
        feature_names = [f"HLA_A_{i:02d}" for i in range(n_features)]

        config = AnalysisConfig(workers=1, chunk_size=5, min_events=3)
        analyzer = SurvivalAnalyzer(config)

        df = analyzer.analyze_stratum(
            dosage, time, event, None, feature_names, "DS1", "overall", "full"
        )

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            expected_cols = {"feature", "beta", "se", "hr", "pvalue", "fdr",
                           "dataset", "stratum", "strategy"}
            assert expected_cols.issubset(set(df.columns))

    def test_few_events_skip(self):
        """Test that stratum with <2 events is skipped."""
        n = 50
        n_features = 5
        time = np.random.exponential(500, n)
        event = np.zeros(n)  # No events
        event[0] = 1  # Only 1 event
        dosage = np.random.choice([0, 1, 2], (n, n_features)).astype(np.float32)
        feature_names = [f"HLA_A_{i:02d}" for i in range(n_features)]

        config = AnalysisConfig(workers=1, chunk_size=5, min_events=1)
        analyzer = SurvivalAnalyzer(config)

        df = analyzer.analyze_stratum(
            dosage, time, event, None, feature_names, "DS1", "overall", "full"
        )
        assert df.empty  # Should skip due to <2 events
