"""
Tests for the conditional association analysis module.
"""

import numpy as np
import pandas as pd
import pytest

from hla_analysis.conditional import (
    conditional_analysis,
    _extract_gene,
    _get_nearby_features,
)


# ── Gene extraction tests ────────────────────────────────────────────────


class TestExtractGene:
    """Test _extract_gene helper."""

    def test_vcf_style(self):
        assert _extract_gene("HLA_A*02:01") == "A"

    def test_normalised_style(self):
        assert _extract_gene("HLA_A_02:01") == "A"

    def test_drb1(self):
        assert _extract_gene("HLA_DRB1*15:01") == "DRB1"

    def test_dpb1_normalised(self):
        assert _extract_gene("HLA_DPB1_04:01") == "DPB1"

    def test_amino_acid_returns_none(self):
        assert _extract_gene("AA_A_9_V") is None

    def test_random_returns_none(self):
        assert _extract_gene("rs12345") is None


class TestGetNearbyFeatures:
    """Test auto-detection of nearby features."""

    def test_same_gene(self):
        features = ["HLA_A_01:01", "HLA_A_02:01", "HLA_A_03:01", "HLA_B_07:02"]
        nearby = _get_nearby_features("HLA_A_01:01", features)
        assert "HLA_A_02:01" in nearby
        assert "HLA_A_03:01" in nearby
        assert "HLA_B_07:02" not in nearby
        assert "HLA_A_01:01" not in nearby  # exclude target

    def test_gene_family_dpb1_includes_dpa1(self):
        features = ["HLA_DPB1_04:01", "HLA_DPB1_02:01", "HLA_DPA1_01:03", "HLA_A_01:01"]
        nearby = _get_nearby_features("HLA_DPB1_04:01", features)
        assert "HLA_DPB1_02:01" in nearby
        assert "HLA_DPA1_01:03" in nearby
        assert "HLA_A_01:01" not in nearby

    def test_gene_family_dqb1_includes_dqa1(self):
        features = ["HLA_DQB1_06:02", "HLA_DQA1_01:02", "HLA_B_08:01"]
        nearby = _get_nearby_features("HLA_DQB1_06:02", features)
        assert "HLA_DQA1_01:02" in nearby
        assert "HLA_B_08:01" not in nearby

    def test_no_nearby(self):
        features = ["HLA_A_01:01", "HLA_B_07:02"]
        nearby = _get_nearby_features("HLA_A_01:01", features)
        assert nearby == []  # no other A alleles

    def test_empty_list(self):
        assert _get_nearby_features("HLA_A_01:01", []) == []


# ── Conditional analysis tests ───────────────────────────────────────────


@pytest.fixture
def synthetic_conditional_data():
    """Create synthetic data where target has a true effect and nearby doesn't."""
    rng = np.random.RandomState(42)
    n = 300

    # Two features at gene A, one at gene B
    feature_names = ["HLA_A_01:01", "HLA_A_02:01", "HLA_B_07:02"]

    # Target (HLA_A_01:01) has a real effect on outcome
    target_dosage = rng.choice([0, 1, 2], n, p=[0.5, 0.4, 0.1])

    # Nearby (HLA_A_02:01) is correlated with target but has no independent effect
    nearby_dosage = np.where(target_dosage == 0, 
                              rng.choice([1, 2], n, p=[0.7, 0.3]),
                              rng.choice([0, 1], n, p=[0.7, 0.3]))

    # Unrelated gene
    other_dosage = rng.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2])

    dosage_matrix = np.column_stack([target_dosage, nearby_dosage, other_dosage]).astype(np.float32)

    # Case/control influenced by target
    logit = -1.0 + 0.8 * target_dosage + rng.normal(0, 0.3, n)
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob).astype(np.float64)

    # Survival data influenced by target
    baseline_time = rng.exponential(1000, n)
    time = baseline_time * np.exp(-0.3 * target_dosage)
    time = np.maximum(time, 1.0)
    event = rng.binomial(1, 0.7, n).astype(np.float64)

    return dosage_matrix, feature_names, y, time, event


class TestConditionalAnalysis:
    """Test conditional_analysis function."""

    def test_risk_returns_dataframe(self, synthetic_conditional_data):
        dosage, features, y, time, event = synthetic_conditional_data
        result = conditional_analysis(
            dosage, features, "HLA_A_01:01",
            y=y, analysis_type="risk",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_risk_expected_columns(self, synthetic_conditional_data):
        dosage, features, y, time, event = synthetic_conditional_data
        result = conditional_analysis(
            dosage, features, "HLA_A_01:01",
            y=y, analysis_type="risk",
        )
        expected_cols = [
            "feature", "target",
            "beta_unconditional", "pvalue_unconditional",
            "beta_conditional", "pvalue_conditional",
            "beta_target_conditional", "pvalue_target_conditional",
            "independence",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_risk_auto_detects_nearby(self, synthetic_conditional_data):
        dosage, features, y, time, event = synthetic_conditional_data
        result = conditional_analysis(
            dosage, features, "HLA_A_01:01",
            y=y, analysis_type="risk",
        )
        # Should only test HLA_A_02:01 (same gene), not HLA_B_07:02
        assert set(result["feature"]) == {"HLA_A_02:01"}

    def test_risk_explicit_nearby(self, synthetic_conditional_data):
        dosage, features, y, time, event = synthetic_conditional_data
        result = conditional_analysis(
            dosage, features, "HLA_A_01:01",
            nearby_features=["HLA_A_02:01", "HLA_B_07:02"],
            y=y, analysis_type="risk",
        )
        assert len(result) == 2

    def test_survival_returns_dataframe(self, synthetic_conditional_data):
        dosage, features, y, time, event = synthetic_conditional_data
        result = conditional_analysis(
            dosage, features, "HLA_A_01:01",
            time=time, event=event, analysis_type="survival",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_target_not_found(self, synthetic_conditional_data):
        dosage, features, y, time, event = synthetic_conditional_data
        result = conditional_analysis(
            dosage, features, "HLA_NONEXIST_99:99",
            y=y, analysis_type="risk",
        )
        assert result.empty

    def test_no_nearby_features(self):
        """Target with no nearby features returns empty."""
        rng = np.random.RandomState(0)
        n = 100
        dosage = rng.choice([0, 1, 2], (n, 2)).astype(np.float32)
        features = ["HLA_A_01:01", "HLA_B_07:02"]  # different genes
        y = rng.binomial(1, 0.5, n).astype(np.float64)
        result = conditional_analysis(dosage, features, "HLA_A_01:01",
                                       y=y, analysis_type="risk")
        assert result.empty

    def test_with_covariates(self, synthetic_conditional_data):
        dosage, features, y, time, event = synthetic_conditional_data
        rng = np.random.RandomState(99)
        X_cov = rng.normal(0, 1, (len(y), 2))
        result = conditional_analysis(
            dosage, features, "HLA_A_01:01",
            y=y, X_cov=X_cov, analysis_type="risk",
        )
        assert len(result) > 0
        assert "beta_conditional" in result.columns

    def test_independence_label(self, synthetic_conditional_data):
        dosage, features, y, time, event = synthetic_conditional_data
        result = conditional_analysis(
            dosage, features, "HLA_A_01:01",
            y=y, analysis_type="risk",
        )
        assert all(r in ["independent", "not_independent", "failed"]
                   for r in result["independence"])


# ── Best-adjusted meta-analysis tests ────────────────────────────────────


class TestBestAdjustedResults:
    """Test create_best_adjusted_results from meta_analysis module."""

    def test_basic_selection(self):
        from hla_analysis.meta_analysis import create_best_adjusted_results

        df = pd.DataFrame({
            "dataset": ["DS1", "DS1", "DS2", "DS2"],
            "feature": ["F1", "F1", "F1", "F1"],
            "stratum": ["overall", "overall", "overall", "overall"],
            "strategy": ["all_covariates", "no_covariates", "all_covariates", "no_covariates"],
            "beta": [0.5, 0.6, 0.4, 0.7],
            "se": [0.1, 0.12, 0.11, 0.13],
            "converged": [True, True, True, True],
        })
        result = create_best_adjusted_results(df)
        assert len(result) == 2  # one per dataset
        assert all(result["strategy"] == "best_adjusted")
        # Should pick all_covariates for both (higher priority)
        assert all(result["source_strategy"] == "all_covariates")

    def test_fallback_when_primary_invalid(self):
        from hla_analysis.meta_analysis import create_best_adjusted_results

        df = pd.DataFrame({
            "dataset": ["DS1", "DS1"],
            "feature": ["F1", "F1"],
            "stratum": ["overall", "overall"],
            "strategy": ["all_covariates", "no_covariates"],
            "beta": [np.nan, 0.6],
            "se": [np.nan, 0.12],
            "converged": [True, True],
        })
        result = create_best_adjusted_results(df)
        assert len(result) == 1
        assert result.iloc[0]["source_strategy"] == "no_covariates"

    def test_empty_input(self):
        from hla_analysis.meta_analysis import create_best_adjusted_results
        result = create_best_adjusted_results(pd.DataFrame())
        assert result.empty

    def test_all_invalid(self):
        from hla_analysis.meta_analysis import create_best_adjusted_results
        df = pd.DataFrame({
            "dataset": ["DS1"],
            "feature": ["F1"],
            "stratum": ["overall"],
            "strategy": ["full"],
            "beta": [np.nan],
            "se": [np.nan],
            "converged": [False],
        })
        result = create_best_adjusted_results(df)
        assert result.empty

    def test_preference_order(self):
        from hla_analysis.meta_analysis import create_best_adjusted_results

        df = pd.DataFrame({
            "dataset": ["DS1", "DS1", "DS1"],
            "feature": ["F1", "F1", "F1"],
            "stratum": ["overall", "overall", "overall"],
            "strategy": ["drop_age", "full", "reduced"],
            "beta": [0.3, 0.5, 0.4],
            "se": [0.1, 0.1, 0.1],
            "converged": [True, True, True],
        })
        result = create_best_adjusted_results(df)
        assert len(result) == 1
        # full > drop_age > reduced
        assert result.iloc[0]["source_strategy"] == "full"

    def test_multiple_features(self):
        from hla_analysis.meta_analysis import create_best_adjusted_results

        df = pd.DataFrame({
            "dataset": ["DS1", "DS1", "DS1", "DS1"],
            "feature": ["F1", "F1", "F2", "F2"],
            "stratum": ["overall", "overall", "overall", "overall"],
            "strategy": ["full", "reduced", "full", "reduced"],
            "beta": [0.5, 0.4, np.nan, 0.3],
            "se": [0.1, 0.1, np.nan, 0.1],
            "converged": [True, True, True, True],
        })
        result = create_best_adjusted_results(df)
        assert len(result) == 2
        f1 = result[result["feature"] == "F1"]
        f2 = result[result["feature"] == "F2"]
        assert f1.iloc[0]["source_strategy"] == "full"
        assert f2.iloc[0]["source_strategy"] == "reduced"
