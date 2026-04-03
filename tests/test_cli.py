"""Tests for cli module."""

import pytest

from hla_analysis.cli import parse_args, build_parser
from hla_analysis.config import AnalysisConfig


class TestBuildParser:
    """Tests for argument parser construction."""

    def test_parser_exists(self):
        parser = build_parser()
        assert parser is not None

    def test_required_args(self):
        """Test that required args are enforced."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # Missing required args


class TestParseArgs:
    """Tests for argument parsing."""

    def test_basic_parse(self, tmp_path):
        """Test parsing minimal arguments."""
        # Create dummy files
        (tmp_path / "dos.csv").touch()
        (tmp_path / "cov.csv").touch()

        config = parse_args([
            "--dosage-files", str(tmp_path / "dos.csv"),
            "--covariate-files", str(tmp_path / "cov.csv"),
        ])
        assert isinstance(config, AnalysisConfig)
        assert len(config.dosage_files) == 1
        assert len(config.covariate_files) == 1

    def test_dataset_names_auto(self, tmp_path):
        """Test auto-derived dataset names."""
        (tmp_path / "cidr_dosage.csv").touch()
        (tmp_path / "cidr_cov.csv").touch()

        config = parse_args([
            "--dosage-files", str(tmp_path / "cidr_dosage.csv"),
            "--covariate-files", str(tmp_path / "cidr_cov.csv"),
        ])
        assert config.dataset_names == ["cidr_dosage"]

    def test_explicit_dataset_names(self, tmp_path):
        """Test explicit dataset names."""
        (tmp_path / "d1.csv").touch()
        (tmp_path / "c1.csv").touch()

        config = parse_args([
            "--dosage-files", str(tmp_path / "d1.csv"),
            "--covariate-files", str(tmp_path / "c1.csv"),
            "--dataset-names", "MyDataset",
        ])
        assert config.dataset_names == ["MyDataset"]

    def test_all_args(self, tmp_path):
        """Test parsing all arguments."""
        for name in ["d1.csv", "d2.csv", "c1.csv", "c2.csv"]:
            (tmp_path / name).touch()

        config = parse_args([
            "--dosage-files", str(tmp_path / "d1.csv"), str(tmp_path / "d2.csv"),
            "--covariate-files", str(tmp_path / "c1.csv"), str(tmp_path / "c2.csv"),
            "--dataset-names", "DS1", "DS2",
            "--output-dir", str(tmp_path / "results"),
            "--analyses", "risk",
            "--strata", "overall", "idh_wt",
            "--risk-covariates", "age", "sex",
            "--survival-covariates", "age", "grade",
            "--covariate-strategies", "full",
            "--min-carriers", "15",
            "--min-events", "8",
            "--missingness-threshold", "0.25",
            "--fdr-threshold", "0.1",
            "--meta-min-datasets", "3",
            "--workers", "4",
            "--memory-limit", "64",
            "--chunk-size", "200",
            "--feature-types", "classical_4digit",
            "--plots", "manhattan",
            "--log-level", "DEBUG",
            "--seed", "123",
            "--cox-solver", "lifelines",
            "--cox-penalizer", "0.05",
        ])

        assert config.analyses == ["risk"]
        assert config.strata == ["overall", "idh_wt"]
        assert config.min_carriers == 15
        assert config.min_events == 8
        assert config.missingness_threshold == 0.25
        assert config.fdr_threshold == 0.1
        assert config.workers == 4
        assert config.memory_limit == 64
        assert config.chunk_size == 200
        assert config.seed == 123
        assert config.cox_solver == "lifelines"

    def test_defaults(self, tmp_path):
        """Test default values."""
        (tmp_path / "d.csv").touch()
        (tmp_path / "c.csv").touch()
        config = parse_args([
            "--dosage-files", str(tmp_path / "d.csv"),
            "--covariate-files", str(tmp_path / "c.csv"),
        ])
        assert config.min_carriers == 10
        assert config.min_events == 5
        assert config.fdr_threshold == 0.05
        assert config.seed == 42
        assert "risk" in config.analyses
        assert "survival" in config.analyses

    def test_sensitivity_flag(self, tmp_path):
        """Test --sensitivity-analysis flag."""
        (tmp_path / "d.csv").touch()
        (tmp_path / "c.csv").touch()
        config = parse_args([
            "--dosage-files", str(tmp_path / "d.csv"),
            "--covariate-files", str(tmp_path / "c.csv"),
            "--sensitivity-analysis",
        ])
        assert config.sensitivity_analysis is True

    def test_sensitivity_flag_default_off(self, tmp_path):
        """Test that sensitivity analysis is off by default."""
        (tmp_path / "d.csv").touch()
        (tmp_path / "c.csv").touch()
        config = parse_args([
            "--dosage-files", str(tmp_path / "d.csv"),
            "--covariate-files", str(tmp_path / "c.csv"),
        ])
        assert config.sensitivity_analysis is False

    def test_sensitivity_short_flag(self, tmp_path):
        """Test --sensitivity short form of the flag."""
        (tmp_path / "d.csv").touch()
        (tmp_path / "c.csv").touch()
        config = parse_args([
            "--dosage-files", str(tmp_path / "d.csv"),
            "--covariate-files", str(tmp_path / "c.csv"),
            "--sensitivity",
        ])
        assert config.sensitivity_analysis is True

    def test_sensitivity_with_plots(self, tmp_path):
        """Test sensitivity plot type in --plots."""
        (tmp_path / "d.csv").touch()
        (tmp_path / "c.csv").touch()
        config = parse_args([
            "--dosage-files", str(tmp_path / "d.csv"),
            "--covariate-files", str(tmp_path / "c.csv"),
            "--sensitivity-analysis",
            "--plots", "manhattan", "sensitivity",
        ])
        assert "sensitivity" in config.plots
        assert config.sensitivity_analysis is True

