"""Tests for meta_analysis module."""

import numpy as np
import pandas as pd
import pytest

from hla_analysis.config import AnalysisConfig
from hla_analysis.meta_analysis import (
    MetaAnalyzer,
    fixed_effects,
    random_effects,
    cochran_q,
    dersimonian_laird_tau2,
    create_summary_tables,
)


class TestFixedEffects:
    """Tests for fixed-effects meta-analysis."""

    def test_basic(self):
        """Test basic fixed-effects calculation."""
        betas = np.array([0.5, 0.3, 0.7])
        ses = np.array([0.1, 0.15, 0.12])

        result = fixed_effects(betas, ses)

        assert "pooled_beta" in result
        assert "pooled_se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "pvalue" in result
        # CI should contain the pooled estimate
        assert result["ci_lower"] < result["pooled_beta"] < result["ci_upper"]

    def test_inverse_variance_weighting(self):
        """Test that weights are proportional to 1/SE²."""
        betas = np.array([1.0, 2.0])
        ses = np.array([0.1, 1.0])  # First study has 100x more weight

        result = fixed_effects(betas, ses)
        # Should be much closer to 1.0 (first study)
        assert abs(result["pooled_beta"] - 1.0) < abs(result["pooled_beta"] - 2.0)

    def test_single_study(self):
        """Test with a single study."""
        betas = np.array([0.5])
        ses = np.array([0.1])

        result = fixed_effects(betas, ses)
        assert result["pooled_beta"] == pytest.approx(0.5, abs=1e-10)
        assert result["pooled_se"] == pytest.approx(0.1, abs=1e-10)

    def test_identical_studies(self):
        """Test with identical studies."""
        betas = np.array([0.5, 0.5, 0.5])
        ses = np.array([0.1, 0.1, 0.1])

        result = fixed_effects(betas, ses)
        assert result["pooled_beta"] == pytest.approx(0.5, abs=1e-10)
        # SE should decrease with more studies
        assert result["pooled_se"] < 0.1

    def test_pooled_se_formula(self):
        """Test pooled SE = 1/sqrt(sum(1/SE²))."""
        betas = np.array([0.5, 0.3])
        ses = np.array([0.1, 0.2])
        weights = 1.0 / ses**2
        expected_se = 1.0 / np.sqrt(weights.sum())

        result = fixed_effects(betas, ses)
        assert result["pooled_se"] == pytest.approx(expected_se, rel=1e-10)


class TestCochranQ:
    """Tests for heterogeneity statistics."""

    def test_no_heterogeneity(self):
        """Test Q=0 and I²=0 when all studies agree."""
        betas = np.array([0.5, 0.5, 0.5])
        ses = np.array([0.1, 0.1, 0.1])
        pooled = 0.5

        result = cochran_q(betas, ses, pooled)
        assert result["q_stat"] == pytest.approx(0.0, abs=1e-10)
        assert result["i_squared"] == pytest.approx(0.0, abs=1e-10)

    def test_high_heterogeneity(self):
        """Test high I² with divergent studies."""
        betas = np.array([0.1, 1.0, -0.5])
        ses = np.array([0.05, 0.05, 0.05])  # Small SEs, large differences
        pooled = (betas / ses**2).sum() / (1/ses**2).sum()

        result = cochran_q(betas, ses, pooled)
        assert result["q_stat"] > 0
        assert result["i_squared"] > 50  # Should show substantial heterogeneity

    def test_i_squared_range(self):
        """Test I² is between 0 and 100."""
        rng = np.random.RandomState(42)
        betas = rng.normal(0.5, 0.3, 5)
        ses = rng.uniform(0.05, 0.2, 5)
        pooled = (betas / ses**2).sum() / (1/ses**2).sum()

        result = cochran_q(betas, ses, pooled)
        assert 0 <= result["i_squared"] <= 100


class TestDerSimonianLaird:
    """Tests for DerSimonian-Laird tau² estimator."""

    def test_zero_heterogeneity(self):
        """Test tau²=0 when no between-study variance."""
        betas = np.array([0.5, 0.5, 0.5])
        ses = np.array([0.1, 0.1, 0.1])

        tau2 = dersimonian_laird_tau2(betas, ses)
        assert tau2 == pytest.approx(0.0, abs=1e-10)

    def test_positive_tau2(self):
        """Test tau² > 0 with heterogeneous studies."""
        betas = np.array([0.1, 1.0, -0.5])
        ses = np.array([0.05, 0.05, 0.05])

        tau2 = dersimonian_laird_tau2(betas, ses)
        assert tau2 > 0

    def test_non_negative(self):
        """Test tau² is never negative."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            betas = rng.normal(0, 0.5, 3)
            ses = rng.uniform(0.05, 0.5, 3)
            tau2 = dersimonian_laird_tau2(betas, ses)
            assert tau2 >= 0


class TestRandomEffects:
    """Tests for random-effects meta-analysis."""

    def test_basic(self):
        """Test random effects returns expected keys."""
        betas = np.array([0.5, 0.3, 0.7])
        ses = np.array([0.1, 0.15, 0.12])

        result = random_effects(betas, ses)
        assert "pooled_beta" in result
        assert "pooled_se" in result
        assert "tau2" in result
        assert result["ci_lower"] < result["pooled_beta"] < result["ci_upper"]

    def test_re_se_geq_fe_se(self):
        """Test that RE SE >= FE SE (incorporating between-study variance)."""
        betas = np.array([0.1, 0.5, 0.9])
        ses = np.array([0.1, 0.1, 0.1])

        fe = fixed_effects(betas, ses)
        re = random_effects(betas, ses)
        # RE SE should be >= FE SE
        assert re["pooled_se"] >= fe["pooled_se"] - 1e-10

    def test_matches_fe_when_no_heterogeneity(self):
        """Test RE matches FE when tau²=0."""
        betas = np.array([0.5, 0.5, 0.5])
        ses = np.array([0.1, 0.1, 0.1])

        fe = fixed_effects(betas, ses)
        re = random_effects(betas, ses)
        assert re["pooled_beta"] == pytest.approx(fe["pooled_beta"], abs=1e-10)


class TestMetaAnalyzer:
    """Tests for the MetaAnalyzer class."""

    def test_run_meta_analysis(self):
        """Test full meta-analysis workflow."""
        df = pd.DataFrame({
            "feature": ["HLA_A_01"]*3 + ["HLA_B_01"]*3,
            "dataset": ["DS1", "DS2", "DS3"]*2,
            "stratum": ["overall"]*6,
            "strategy": ["full"]*6,
            "beta": [0.3, 0.5, 0.4, -0.2, -0.3, -0.1],
            "se": [0.1, 0.12, 0.11, 0.15, 0.14, 0.16],
            "pvalue": [0.003, 0.001, 0.002, 0.18, 0.03, 0.53],
            "converged": [True]*6,
        })

        config = AnalysisConfig(workers=1, meta_min_datasets=2)
        analyzer = MetaAnalyzer(config)
        result = analyzer.run_meta_analysis(df, analysis_type="risk")

        assert len(result) == 2  # Two features
        assert "fe_beta" in result.columns
        assert "re_beta" in result.columns
        assert "i_squared" in result.columns
        assert "fe_fdr" in result.columns

    def test_min_datasets_filter(self):
        """Test that features with too few datasets are excluded."""
        df = pd.DataFrame({
            "feature": ["HLA_A_01"],
            "dataset": ["DS1"],
            "stratum": ["overall"],
            "strategy": ["full"],
            "beta": [0.3],
            "se": [0.1],
            "pvalue": [0.003],
            "converged": [True],
        })

        config = AnalysisConfig(workers=1, meta_min_datasets=2)
        analyzer = MetaAnalyzer(config)
        result = analyzer.run_meta_analysis(df, analysis_type="risk")
        assert result.empty

    def test_non_converged_excluded(self):
        """Test that non-converged results are excluded."""
        df = pd.DataFrame({
            "feature": ["HLA_A_01"]*2,
            "dataset": ["DS1", "DS2"],
            "stratum": ["overall"]*2,
            "strategy": ["full"]*2,
            "beta": [0.3, np.nan],
            "se": [0.1, np.nan],
            "pvalue": [0.003, np.nan],
            "converged": [True, False],
        })

        config = AnalysisConfig(workers=1, meta_min_datasets=2)
        analyzer = MetaAnalyzer(config)
        result = analyzer.run_meta_analysis(df, analysis_type="risk")
        assert result.empty  # Only 1 valid dataset


class TestCreateSummaryTables:
    """Tests for summary table creation."""

    def test_empty_inputs(self):
        """Test with empty DataFrames."""
        tables = create_summary_tables(pd.DataFrame(), pd.DataFrame())
        assert "risk_significant" in tables
        assert tables["risk_significant"].empty

    def test_with_data(self):
        """Test summary tables with mock meta-analysis results."""
        risk_meta = pd.DataFrame({
            "feature": ["HLA_A_01", "HLA_B_01"],
            "stratum": ["overall", "overall"],
            "strategy": ["full", "full"],
            "fe_pvalue": [0.001, 0.5],
            "fe_fdr": [0.002, 0.5],
            "fe_beta": [0.5, 0.1],
        })
        tables = create_summary_tables(risk_meta, pd.DataFrame(), fdr_threshold=0.05)
        assert len(tables["risk_significant"]) == 1
