"""Tests for visualization module."""

import os

import numpy as np
import pandas as pd
import pytest

from hla_analysis.visualization import Visualizer


@pytest.fixture
def viz_dir(tmp_path):
    """Temporary directory for plot output."""
    d = tmp_path / "plots"
    d.mkdir()
    return str(d)


@pytest.fixture
def sample_meta_df():
    """Sample meta-analysis DataFrame for plotting."""
    features = [f"HLA_A_{i:02d}" for i in range(10)] + [f"AA_B_{i}_V" for i in range(5)]
    n = len(features)
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "feature": features,
        "stratum": ["overall"] * n,
        "strategy": ["full"] * n,
        "fe_pvalue": rng.uniform(1e-6, 0.5, n),
        "fe_beta": rng.normal(0, 0.5, n),
        "fe_or": np.exp(rng.normal(0, 0.5, n)),
        "fe_or_lower": np.exp(rng.normal(-0.5, 0.3, n)),
        "fe_or_upper": np.exp(rng.normal(0.5, 0.3, n)),
        "fe_fdr": rng.uniform(1e-4, 0.5, n),
        "fe_hr": np.exp(rng.normal(0, 0.5, n)),
        "fe_hr_lower": np.exp(rng.normal(-0.5, 0.3, n)),
        "fe_hr_upper": np.exp(rng.normal(0.5, 0.3, n)),
    })


@pytest.fixture
def sample_per_dataset_df():
    """Sample per-dataset DataFrame for forest plots."""
    features = [f"HLA_A_{i:02d}" for i in range(10)]
    rows = []
    rng = np.random.RandomState(42)
    for feat in features:
        for ds in ["DS1", "DS2"]:
            rows.append({
                "feature": feat,
                "dataset": ds,
                "stratum": "overall",
                "strategy": "full",
                "or_val": float(np.exp(rng.normal(0, 0.5))),
                "hr": float(np.exp(rng.normal(0, 0.5))),
                "ci_lower": float(np.exp(rng.normal(-0.5, 0.3))),
                "ci_upper": float(np.exp(rng.normal(0.5, 0.3))),
                "pvalue": float(rng.uniform(1e-4, 0.5)),
            })
    return pd.DataFrame(rows)


class TestVisualizer:
    """Tests for Visualizer class."""

    def test_manhattan_plot(self, viz_dir, sample_meta_df):
        """Test Manhattan plot generation."""
        viz = Visualizer(output_dir=viz_dir)
        path = viz.manhattan_plot(sample_meta_df, "risk", "overall", "full")
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_manhattan_plot_empty(self, viz_dir):
        """Test Manhattan plot with empty data."""
        viz = Visualizer(output_dir=viz_dir)
        path = viz.manhattan_plot(pd.DataFrame(), "risk", "overall", "full")
        assert path == ""

    def test_forest_plot(self, viz_dir, sample_meta_df, sample_per_dataset_df):
        """Test forest plot generation."""
        viz = Visualizer(output_dir=viz_dir)
        path = viz.forest_plot(
            sample_meta_df, sample_per_dataset_df,
            "risk", "overall", "full", top_n=5,
        )
        assert os.path.exists(path)

    def test_heatmap_plot(self, viz_dir):
        """Test heatmap plot generation."""
        features = [f"HLA_A_{i:02d}" for i in range(5)]
        strata = ["overall", "idh_wt", "idh_mut"]
        rows = []
        rng = np.random.RandomState(42)
        for feat in features:
            for s in strata:
                rows.append({
                    "feature": feat,
                    "stratum": s,
                    "strategy": "full",
                    "fe_beta": float(rng.normal(0, 0.5)),
                    "fe_pvalue": float(rng.uniform(1e-4, 0.5)),
                })
        df = pd.DataFrame(rows)

        viz = Visualizer(output_dir=viz_dir)
        path = viz.heatmap_plot(df, "risk", "full", top_n=5)
        assert os.path.exists(path)

    def test_comparison_plot(self, viz_dir):
        """Test FULL vs REDUCED comparison plot."""
        rng = np.random.RandomState(42)
        n = 20
        rows = []
        for i in range(n):
            for strat in ["full", "reduced"]:
                rows.append({
                    "feature": f"HLA_A_{i:02d}",
                    "stratum": "overall",
                    "strategy": strat,
                    "fe_pvalue": float(rng.uniform(1e-4, 0.5)),
                })
        df = pd.DataFrame(rows)

        viz = Visualizer(output_dir=viz_dir)
        path = viz.comparison_plot(df, "risk")
        assert os.path.exists(path)

    def test_comparison_plot_single_strategy(self, viz_dir, sample_meta_df):
        """Test comparison plot with only one strategy returns empty."""
        viz = Visualizer(output_dir=viz_dir)
        path = viz.comparison_plot(sample_meta_df, "risk")
        # Only "full" strategy => should return empty
        assert path == ""

    def test_sensitivity_plot(self, viz_dir):
        """Test sensitivity comparison scatter plot."""
        rng = np.random.RandomState(42)
        n = 30
        df = pd.DataFrame({
            "feature": [f"HLA_A_{i:02d}" for i in range(n)],
            "stratum": ["overall"] * n,
            "dataset": ["DS1"] * n,
            "all_covariates_pvalue": rng.uniform(1e-5, 0.5, n),
            "drop_age_pvalue": rng.uniform(1e-5, 0.5, n),
            "no_covariates_pvalue": rng.uniform(1e-5, 0.5, n),
            "all_covariates_beta": rng.normal(0, 0.5, n),
            "drop_age_beta": rng.normal(0, 0.5, n),
            "no_covariates_beta": rng.normal(0, 0.5, n),
        })

        viz = Visualizer(output_dir=viz_dir)
        path = viz.sensitivity_plot(df, "risk")
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_sensitivity_plot_empty(self, viz_dir):
        """Test sensitivity plot with empty data."""
        viz = Visualizer(output_dir=viz_dir)
        path = viz.sensitivity_plot(pd.DataFrame(), "risk")
        assert path == ""

    def test_sensitivity_plot_no_alternatives(self, viz_dir):
        """Test sensitivity plot with only baseline column."""
        df = pd.DataFrame({
            "feature": ["HLA_A_01"],
            "all_covariates_pvalue": [0.01],
        })
        viz = Visualizer(output_dir=viz_dir)
        path = viz.sensitivity_plot(df, "risk")
        assert path == ""

