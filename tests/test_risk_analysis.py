"""Tests for risk_analysis module."""

import numpy as np
import pandas as pd
import pytest

from hla_analysis.config import AnalysisConfig
from hla_analysis.risk_analysis import RiskAnalyzer, _fit_logistic_single


class TestFitLogisticSingle:
    """Tests for the single-feature logistic regression fitter."""

    def test_basic_fit(self):
        """Test basic logistic regression fit returns expected keys."""
        rng = np.random.RandomState(42)
        n = 200
        y = np.array([1]*100 + [0]*100, dtype=float)
        dosage = rng.choice([0, 1, 2], n, p=[0.5, 0.35, 0.15]).astype(float)
        # Add some signal
        dosage[:100] += 0.3

        result = _fit_logistic_single(dosage, y, None, "test_feature", min_carriers=5)

        assert result is not None
        assert "beta" in result
        assert "se" in result
        assert "or_val" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "pvalue" in result
        assert result["converged"] is True
        assert result["or_val"] == pytest.approx(np.exp(result["beta"]), rel=1e-3)

    def test_with_covariates(self):
        """Test logistic regression with covariates."""
        rng = np.random.RandomState(42)
        n = 200
        y = np.array([1]*100 + [0]*100, dtype=float)
        dosage = rng.choice([0, 1, 2], n).astype(float)
        X_cov = rng.normal(0, 1, (n, 3))

        result = _fit_logistic_single(dosage, y, X_cov, "test_feature", min_carriers=5)
        assert result is not None
        assert result["n_cases"] == 100
        assert result["n_controls"] == 100

    def test_or_ci_correct(self):
        """Test that OR and CI are correctly computed from beta and SE."""
        rng = np.random.RandomState(42)
        n = 300
        y = np.array([1]*150 + [0]*150, dtype=float)
        dosage = rng.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2]).astype(float)
        dosage[:150] += 0.5

        result = _fit_logistic_single(dosage, y, None, "test", min_carriers=5)
        if result and result["converged"]:
            assert result["or_val"] == pytest.approx(np.exp(result["beta"]), rel=1e-5)
            assert result["ci_lower"] == pytest.approx(
                np.exp(result["beta"] - 1.96 * result["se"]), rel=1e-3
            )
            assert result["ci_upper"] == pytest.approx(
                np.exp(result["beta"] + 1.96 * result["se"]), rel=1e-3
            )

    def test_skip_low_carrier(self):
        """Test that features with too few carriers are skipped."""
        n = 100
        y = np.array([1]*50 + [0]*50, dtype=float)
        dosage = np.zeros(n)  # No carriers
        dosage[0] = 1  # Only 1 carrier in cases

        result = _fit_logistic_single(dosage, y, None, "rare_feature", min_carriers=10)
        assert result is None

    def test_skip_zero_variance(self):
        """Test that zero-variance features are skipped."""
        n = 100
        y = np.array([1]*50 + [0]*50, dtype=float)
        dosage = np.ones(n)  # All same value

        result = _fit_logistic_single(dosage, y, None, "constant", min_carriers=1)
        assert result is None

    def test_perfect_separation(self):
        """Test handling of near-perfect separation (should not crash)."""
        rng = np.random.RandomState(42)
        n = 200
        y = np.array([1]*100 + [0]*100, dtype=float)
        # Near-perfect separation: most cases have dosage 2, most controls 0
        dosage = np.concatenate([
            rng.choice([1, 2], 100, p=[0.1, 0.9]),
            rng.choice([0, 1], 100, p=[0.9, 0.1]),
        ]).astype(float)

        result = _fit_logistic_single(dosage, y, None, "perfect_sep", min_carriers=5)
        # Should return something (either converged with extreme values or not converged)
        assert result is not None


class TestRiskAnalyzer:
    """Tests for the RiskAnalyzer class."""

    def test_analyze_stratum(self):
        """Test analyzing a stratum returns a DataFrame with expected columns."""
        rng = np.random.RandomState(42)
        n = 200
        n_features = 10
        y = np.array([1]*100 + [0]*100, dtype=float)
        dosage = rng.choice([0, 1, 2], (n, n_features)).astype(np.float32)
        feature_names = [f"HLA_A_{i:02d}" for i in range(n_features)]

        config = AnalysisConfig(workers=1, chunk_size=5, min_carriers=5)
        analyzer = RiskAnalyzer(config)

        df = analyzer.analyze_stratum(
            dosage, y, None, feature_names, "test_ds", "overall", "full"
        )

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            expected_cols = {"feature", "beta", "se", "or_val", "pvalue", "fdr",
                           "dataset", "stratum", "strategy"}
            assert expected_cols.issubset(set(df.columns))

    def test_analyze_with_covariates(self):
        """Test risk analysis with covariates."""
        rng = np.random.RandomState(42)
        n = 200
        n_features = 5
        y = np.array([1]*100 + [0]*100, dtype=float)
        dosage = rng.choice([0, 1, 2], (n, n_features)).astype(np.float32)
        X_cov = rng.normal(0, 1, (n, 2))
        feature_names = [f"HLA_A_{i:02d}" for i in range(n_features)]

        config = AnalysisConfig(workers=1, chunk_size=5, min_carriers=5)
        analyzer = RiskAnalyzer(config)

        df = analyzer.analyze_stratum(
            dosage, y, X_cov, feature_names, "DS1", "overall", "full",
            covariate_names=["age", "sex"],
        )
        assert isinstance(df, pd.DataFrame)

    def test_fdr_applied(self):
        """Test that FDR correction is applied."""
        rng = np.random.RandomState(42)
        n = 300
        n_features = 15
        y = np.array([1]*150 + [0]*150, dtype=float)
        dosage = rng.choice([0, 1, 2], (n, n_features), p=[0.4, 0.4, 0.2]).astype(np.float32)
        feature_names = [f"HLA_A_{i:02d}" for i in range(n_features)]

        config = AnalysisConfig(workers=1, chunk_size=50, min_carriers=5)
        analyzer = RiskAnalyzer(config)
        df = analyzer.analyze_stratum(dosage, y, None, feature_names, "DS1", "overall", "full")

        if not df.empty:
            assert "fdr" in df.columns
            # FDR values should be >= p-values
            valid = df.dropna(subset=["pvalue", "fdr"])
            if not valid.empty:
                assert (valid["fdr"] >= valid["pvalue"] - 1e-10).all()

    def test_run_single_model(self):
        """Test convenience method for single model."""
        rng = np.random.RandomState(42)
        n = 200
        y = np.array([1]*100 + [0]*100, dtype=float)
        dosage = rng.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2]).astype(float)

        config = AnalysisConfig(workers=1, min_carriers=5)
        analyzer = RiskAnalyzer(config)
        result = analyzer.run_single_model(dosage, y, None, "test")
        assert result is not None
