"""Tests for sensitivity analysis module."""

import numpy as np
import pandas as pd
import pytest

from hla_analysis.config import AnalysisConfig, SensitivityStrategy
from hla_analysis.sensitivity import create_sensitivity_comparison, summarise_sensitivity


class TestSensitivityStrategy:
    """Tests for SensitivityStrategy dataclass and generation."""

    def test_generate_no_missingness(self):
        """When no covariates have missingness, only all_covariates and no_covariates."""
        config = AnalysisConfig(workers=1)
        covs = ["age", "sex", "PC1"]
        miss = {"age": 0.0, "sex": 0.0, "PC1": 0.0}

        strategies = config.generate_sensitivity_strategies(covs, miss)
        names = [s.name for s in strategies]

        assert "all_covariates" in names
        assert "no_covariates" in names
        # No drop_* strategies since nothing is missing
        assert not any(n.startswith("drop_") for n in names)

    def test_generate_with_missingness(self):
        """When covariates have missingness, drop_X strategies are created."""
        config = AnalysisConfig(workers=1)
        covs = ["age", "sex", "PC1"]
        miss = {"age": 0.15, "sex": 0.0, "PC1": 0.02}

        strategies = config.generate_sensitivity_strategies(covs, miss)
        names = [s.name for s in strategies]

        assert "all_covariates" in names
        assert "drop_age" in names
        assert "drop_PC1" in names
        assert "drop_sex" not in names  # sex has 0% missing
        assert "no_covariates" in names

    def test_drop_strategy_excludes_correct_covariate(self):
        """Verify that drop_age strategy excludes age but keeps others."""
        config = AnalysisConfig(workers=1)
        covs = ["age", "sex", "grade"]
        miss = {"age": 0.20, "sex": 0.0, "grade": 0.10}

        strategies = config.generate_sensitivity_strategies(covs, miss)
        drop_age = [s for s in strategies if s.name == "drop_age"][0]
        drop_grade = [s for s in strategies if s.name == "drop_grade"][0]

        assert "age" not in drop_age.covariates
        assert "sex" in drop_age.covariates
        assert "grade" in drop_age.covariates

        assert "grade" not in drop_grade.covariates
        assert "age" in drop_grade.covariates

    def test_all_covariates_includes_all(self):
        """The all_covariates strategy should include every covariate."""
        config = AnalysisConfig(workers=1)
        covs = ["age", "sex", "PC1", "PC2"]
        miss = {"age": 0.1, "sex": 0.0, "PC1": 0.0, "PC2": 0.05}

        strategies = config.generate_sensitivity_strategies(covs, miss)
        all_cov = [s for s in strategies if s.name == "all_covariates"][0]
        assert all_cov.covariates == covs

    def test_no_covariates_is_empty(self):
        """The no_covariates strategy should have an empty covariate list."""
        config = AnalysisConfig(workers=1)
        strategies = config.generate_sensitivity_strategies(["age"], {"age": 0.1})
        no_cov = [s for s in strategies if s.name == "no_covariates"][0]
        assert no_cov.covariates == []

    def test_strategies_order(self):
        """Strategies should be ordered: all_covariates, drop_*, no_covariates."""
        config = AnalysisConfig(workers=1)
        covs = ["age", "sex"]
        miss = {"age": 0.1, "sex": 0.05}

        strategies = config.generate_sensitivity_strategies(covs, miss)
        names = [s.name for s in strategies]

        assert names[0] == "all_covariates"
        assert names[-1] == "no_covariates"
        assert all(n.startswith("drop_") for n in names[1:-1])


class TestCreateSensitivityComparison:
    """Tests for the sensitivity comparison table creation."""

    def test_basic_comparison(self):
        """Test basic sensitivity comparison with two strategies."""
        df = pd.DataFrame({
            "feature": ["HLA_A_01", "HLA_A_01", "HLA_B_01", "HLA_B_01"],
            "stratum": ["overall"] * 4,
            "dataset": ["DS1"] * 4,
            "strategy": ["all_covariates", "drop_age", "all_covariates", "drop_age"],
            "beta": [0.5, 0.45, -0.3, -0.28],
            "se": [0.1, 0.09, 0.15, 0.14],
            "pvalue": [1e-5, 2e-5, 0.05, 0.04],
            "or_val": [1.65, 1.57, 0.74, 0.76],
        })

        result = create_sensitivity_comparison(df, analysis_type="risk")

        assert not result.empty
        assert "all_covariates_beta" in result.columns
        assert "drop_age_beta" in result.columns
        assert "all_covariates_pvalue" in result.columns
        assert "drop_age_pvalue" in result.columns
        assert "max_beta_range" in result.columns

    def test_empty_input(self):
        """Test with empty DataFrame."""
        result = create_sensitivity_comparison(pd.DataFrame())
        assert result.empty

    def test_single_strategy(self):
        """Test with only one strategy returns empty."""
        df = pd.DataFrame({
            "feature": ["HLA_A_01"],
            "stratum": ["overall"],
            "dataset": ["DS1"],
            "strategy": ["all_covariates"],
            "beta": [0.5],
            "se": [0.1],
            "pvalue": [0.01],
        })
        result = create_sensitivity_comparison(df)
        assert result.empty

    def test_max_beta_range(self):
        """Test max_beta_range computation."""
        df = pd.DataFrame({
            "feature": ["HLA_A_01"] * 3,
            "stratum": ["overall"] * 3,
            "dataset": ["DS1"] * 3,
            "strategy": ["all_covariates", "drop_age", "no_covariates"],
            "beta": [0.5, 0.3, 0.8],
            "se": [0.1, 0.1, 0.1],
            "pvalue": [0.01, 0.02, 0.005],
        })
        result = create_sensitivity_comparison(df)
        assert not result.empty
        assert result["max_beta_range"].iloc[0] == pytest.approx(0.5, abs=1e-10)

    def test_sorted_by_baseline_pvalue(self):
        """Test that output is sorted by all_covariates_pvalue."""
        df = pd.DataFrame({
            "feature": ["HLA_B_01", "HLA_B_01", "HLA_A_01", "HLA_A_01"],
            "stratum": ["overall"] * 4,
            "dataset": ["DS1"] * 4,
            "strategy": ["all_covariates", "drop_age", "all_covariates", "drop_age"],
            "beta": [0.1, 0.1, 0.5, 0.5],
            "se": [0.1, 0.1, 0.1, 0.1],
            "pvalue": [0.5, 0.4, 0.001, 0.002],
        })
        result = create_sensitivity_comparison(df)
        assert result.iloc[0]["feature"] == "HLA_A_01"  # Lower p-value first


class TestSummariseSensitivity:
    """Tests for summarise_sensitivity."""

    def test_detects_flips(self):
        """Test that features whose significance changes are flagged."""
        df = pd.DataFrame({
            "feature": ["HLA_A_01", "HLA_B_01"],
            "stratum": ["overall", "overall"],
            "dataset": ["DS1", "DS1"],
            "all_covariates_pvalue": [0.001, 0.5],
            "drop_age_pvalue": [0.1, 0.4],
            "all_covariates_beta": [0.5, 0.1],
            "drop_age_beta": [0.3, 0.12],
        })
        result = summarise_sensitivity(
            df, strategies=["all_covariates", "drop_age"], pvalue_threshold=0.05
        )
        # HLA_A_01 flips: sig in all_cov (0.001 < 0.05), not in drop_age (0.1 > 0.05)
        # HLA_B_01 stays non-sig in both
        assert len(result) == 1
        assert result.iloc[0]["feature"] == "HLA_A_01"
        assert "sig_strategies" in result.columns

    def test_empty_input(self):
        """Test with empty DataFrame."""
        result = summarise_sensitivity(pd.DataFrame(), ["a", "b"])
        assert result.empty
