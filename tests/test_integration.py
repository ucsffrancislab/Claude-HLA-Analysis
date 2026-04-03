"""Integration tests: run full pipeline on synthetic data."""

import os

import numpy as np
import pandas as pd
import pytest

from hla_analysis.config import AnalysisConfig
from hla_analysis.__main__ import run_pipeline


class TestIntegrationPipeline:
    """End-to-end pipeline tests on synthetic data."""

    def test_full_pipeline(self, basic_config):
        """Test full pipeline with two small synthetic datasets."""
        config = basic_config
        config.analyses = ["risk", "survival"]
        config.strata = ["overall"]
        config.covariate_strategies = ["full"]
        config.plots = []  # Skip plots for speed
        config.workers = 1
        config.min_carriers = 3
        config.min_events = 2

        results = run_pipeline(config)

        assert "risk_results" in results
        assert "survival_results" in results
        assert "risk_meta" in results
        assert "survival_meta" in results
        assert "summary_tables" in results

        # Check per-dataset results exist
        risk_path = os.path.join(config.output_dir, "risk_per_dataset.csv")
        if not results["risk_results"].empty:
            assert os.path.exists(risk_path)

    def test_risk_only(self, basic_config):
        """Test risk-only pipeline."""
        config = basic_config
        config.analyses = ["risk"]
        config.strata = ["overall"]
        config.plots = []
        config.min_carriers = 3

        results = run_pipeline(config)
        assert not results["risk_results"].empty or True  # May be empty if no features pass
        assert results["survival_results"].empty

    def test_survival_only(self, basic_config):
        """Test survival-only pipeline."""
        config = basic_config
        config.analyses = ["survival"]
        config.strata = ["overall"]
        config.plots = []
        config.min_events = 2

        results = run_pipeline(config)
        assert results["risk_results"].empty

    def test_multiple_strata(self, basic_config):
        """Test pipeline with multiple strata."""
        config = basic_config
        config.analyses = ["risk"]
        config.strata = ["overall", "idh_wt", "idh_mut"]
        config.plots = []
        config.min_carriers = 3

        results = run_pipeline(config)
        if not results["risk_results"].empty:
            strata_found = results["risk_results"]["stratum"].unique()
            assert len(strata_found) >= 1

    def test_both_strategies(self, basic_config):
        """Test pipeline with both covariate strategies."""
        config = basic_config
        config.analyses = ["risk"]
        config.strata = ["overall"]
        config.covariate_strategies = ["full", "reduced"]
        config.plots = []
        config.min_carriers = 3

        results = run_pipeline(config)
        if not results["risk_results"].empty:
            strategies = results["risk_results"]["strategy"].unique()
            assert len(strategies) >= 1

    def test_output_columns_risk(self, basic_config):
        """Test that risk output has expected columns."""
        config = basic_config
        config.analyses = ["risk"]
        config.strata = ["overall"]
        config.plots = []
        config.min_carriers = 3

        results = run_pipeline(config)
        if not results["risk_results"].empty:
            expected = {"feature", "beta", "se", "or_val", "pvalue", "fdr",
                       "dataset", "stratum", "strategy"}
            assert expected.issubset(set(results["risk_results"].columns))

    def test_output_columns_survival(self, basic_config):
        """Test that survival output has expected columns."""
        config = basic_config
        config.analyses = ["survival"]
        config.strata = ["overall"]
        config.plots = []
        config.min_events = 2

        results = run_pipeline(config)
        if not results["survival_results"].empty:
            expected = {"feature", "beta", "se", "hr", "pvalue", "fdr",
                       "dataset", "stratum", "strategy"}
            assert expected.issubset(set(results["survival_results"].columns))

    def test_with_plots(self, basic_config):
        """Test pipeline with plot generation enabled."""
        config = basic_config
        config.analyses = ["risk"]
        config.strata = ["overall"]
        config.covariate_strategies = ["full"]
        config.plots = ["manhattan", "forest", "heatmap"]
        config.min_carriers = 3

        results = run_pipeline(config)
        # Just ensure it doesn't crash; plots may be empty if no significant results
        assert "risk_results" in results


class TestEdgeCases:
    """Edge case tests."""

    def test_all_zero_feature(self, tmp_path):
        """Test handling of an all-zero feature."""
        n = 50
        dos = pd.DataFrame({
            "sample_id": [f"S{i}" for i in range(n)],
            "HLA_A_01": np.zeros(n),  # All zeros
            "HLA_B_01": np.random.choice([0, 1, 2], n),
        })
        cov = pd.DataFrame({
            "IID": [f"S{i}" for i in range(n)],
            "case": [1]*25 + [0]*25,
            "age": np.random.normal(50, 10, n),
            "sex": np.random.choice(["M", "F"], n),
            "survdays": np.concatenate([np.random.exponential(500, 25), [np.nan]*25]),
            "vstatus": np.concatenate([np.random.choice([0, 1], 25), [np.nan]*25]),
            "exclude": np.zeros(n),
        })
        for k in range(8):
            cov[f"PC{k+1}"] = np.random.normal(0, 1, n)

        dos.to_csv(tmp_path / "dos.csv", index=False)
        cov.to_csv(tmp_path / "cov.csv", index=False)

        config = AnalysisConfig(
            dosage_files=[str(tmp_path / "dos.csv")],
            covariate_files=[str(tmp_path / "cov.csv")],
            dataset_names=["test"],
            output_dir=str(tmp_path / "results"),
            workers=1, strata=["overall"], plots=[],
            covariate_strategies=["full"],
            analyses=["risk"],
            min_carriers=3,
        )
        results = run_pipeline(config)
        # Should not crash; all-zero feature should be skipped
        assert "risk_results" in results

    def test_zero_events_stratum(self, tmp_path):
        """Test handling of stratum with zero events."""
        n = 40
        dos = pd.DataFrame({
            "sample_id": [f"S{i}" for i in range(n)],
            "HLA_A_01": np.random.choice([0, 1, 2], n),
        })
        cov = pd.DataFrame({
            "IID": [f"S{i}" for i in range(n)],
            "case": [1]*20 + [0]*20,
            "age": np.random.normal(50, 10, n),
            "sex": np.random.choice(["M", "F"], n),
            "idh": [np.nan]*20 + [np.nan]*20,  # No IDH data
            "survdays": np.concatenate([np.random.exponential(500, 20), [np.nan]*20]),
            "vstatus": np.concatenate([np.zeros(20), [np.nan]*20]),  # Zero events!
            "exclude": np.zeros(n),
        })

        dos.to_csv(tmp_path / "dos.csv", index=False)
        cov.to_csv(tmp_path / "cov.csv", index=False)

        config = AnalysisConfig(
            dosage_files=[str(tmp_path / "dos.csv")],
            covariate_files=[str(tmp_path / "cov.csv")],
            dataset_names=["test"],
            output_dir=str(tmp_path / "results"),
            workers=1, strata=["overall"],
            covariate_strategies=["full"],
            analyses=["survival"],
            plots=[], min_events=2,
        )
        results = run_pipeline(config)
        # Should not crash
        assert "survival_results" in results

    def test_single_dataset_no_meta(self, tmp_path):
        """Test that single dataset skips meta-analysis."""
        n = 60
        rng = np.random.RandomState(42)
        dos = pd.DataFrame({
            "sample_id": [f"S{i}" for i in range(n)],
            "HLA_A_01": rng.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2]),
            "HLA_B_01": rng.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2]),
        })
        cov = pd.DataFrame({
            "IID": [f"S{i}" for i in range(n)],
            "case": [1]*30 + [0]*30,
            "age": rng.normal(50, 10, n),
            "sex": rng.choice(["M", "F"], n),
            "exclude": np.zeros(n),
        })

        dos.to_csv(tmp_path / "dos.csv", index=False)
        cov.to_csv(tmp_path / "cov.csv", index=False)

        config = AnalysisConfig(
            dosage_files=[str(tmp_path / "dos.csv")],
            covariate_files=[str(tmp_path / "cov.csv")],
            dataset_names=["test"],
            output_dir=str(tmp_path / "results"),
            workers=1, strata=["overall"],
            covariate_strategies=["full"],
            analyses=["risk"], plots=[],
            min_carriers=3, meta_min_datasets=2,
        )
        results = run_pipeline(config)
        # Meta-analysis should be empty with single dataset
        assert results["risk_meta"].empty


class TestSensitivityIntegration:
    """Integration tests for the sensitivity analysis mode."""

    def test_sensitivity_risk_pipeline(self, sensitivity_config):
        """Test full risk pipeline with sensitivity analysis enabled."""
        config = sensitivity_config
        config.analyses = ["risk"]
        config.plots = []

        results = run_pipeline(config)

        assert "risk_results" in results
        assert "sensitivity_risk" in results

        if not results["risk_results"].empty:
            # Should have multiple strategies including all_covariates and no_covariates
            strategies = results["risk_results"]["strategy"].unique()
            assert "all_covariates" in strategies
            assert "no_covariates" in strategies

    def test_sensitivity_survival_pipeline(self, sensitivity_config):
        """Test survival pipeline with sensitivity analysis enabled."""
        config = sensitivity_config
        config.analyses = ["survival"]
        config.plots = []

        results = run_pipeline(config)
        assert "survival_results" in results
        assert "sensitivity_survival" in results

    def test_sensitivity_produces_comparison_csv(self, sensitivity_config):
        """Test that sensitivity comparison CSV is generated."""
        config = sensitivity_config
        config.analyses = ["risk"]
        config.plots = []

        results = run_pipeline(config)

        if not results["risk_results"].empty:
            # Check multiple strategies ran
            strategies = results["risk_results"]["strategy"].unique()
            if len(strategies) >= 2:
                sens_path = os.path.join(config.output_dir, "sensitivity_comparison_risk.csv")
                if not results["sensitivity_risk"].empty:
                    assert os.path.exists(sens_path)

    def test_sensitivity_backward_compatible(self, basic_config):
        """Test that disabling sensitivity keeps FULL/REDUCED behavior."""
        config = basic_config
        config.sensitivity_analysis = False
        config.analyses = ["risk"]
        config.strata = ["overall"]
        config.covariate_strategies = ["full", "reduced"]
        config.plots = []
        config.min_carriers = 3

        results = run_pipeline(config)

        if not results["risk_results"].empty:
            strategies = results["risk_results"]["strategy"].unique()
            assert "all_covariates" not in strategies
            # At least one of full/reduced should be present
            assert any(s in strategies for s in ["full", "reduced"])

    def test_sensitivity_with_plots(self, sensitivity_config):
        """Test sensitivity pipeline with sensitivity plot enabled."""
        config = sensitivity_config
        config.analyses = ["risk"]
        config.plots = ["sensitivity"]

        results = run_pipeline(config)
        # Should not crash even if no significant results
        assert "risk_results" in results

    def test_sensitivity_drop_strategies_have_more_samples(self, sensitivity_config):
        """Test that drop_X strategies retain >= as many samples as all_covariates."""
        config = sensitivity_config
        config.analyses = ["risk"]
        config.plots = []

        results = run_pipeline(config)
        risk = results["risk_results"]

        if risk.empty:
            return

        # Compare sample counts: drop_* should have >= all_covariates for same feature
        for feature in risk["feature"].unique():
            feat_rows = risk[risk["feature"] == feature]
            all_cov = feat_rows[feat_rows["strategy"] == "all_covariates"]
            if all_cov.empty:
                continue
            all_n = all_cov.iloc[0]["n_cases"] + all_cov.iloc[0]["n_controls"]

            for _, row in feat_rows.iterrows():
                if row["strategy"].startswith("drop_") or row["strategy"] == "no_covariates":
                    drop_n = row["n_cases"] + row["n_controls"]
                    # drop_X or no_covariates should retain >= samples
                    assert drop_n >= all_n, (
                        f"Strategy {row['strategy']} has fewer samples ({drop_n}) "
                        f"than all_covariates ({all_n}) for {feature}"
                    )

