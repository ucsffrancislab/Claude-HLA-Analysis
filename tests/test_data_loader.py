"""Tests for data_loader module."""

import numpy as np
import pandas as pd
import pytest

from hla_analysis.config import AnalysisConfig
from hla_analysis.data_loader import DataLoader, find_common_features


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_dataset(self, basic_config, tmp_data_dir):
        """Test loading a single dataset."""
        loader = DataLoader(basic_config)
        ds = loader.load_dataset(
            str(tmp_data_dir / "ds1_dosage.csv"),
            str(tmp_data_dir / "ds1_cov.csv"),
            "DS1",
        )
        assert "covariates" in ds
        assert "dosage" in ds
        assert "feature_names" in ds
        assert ds["dosage"].dtype == np.float32
        assert ds["dosage"].shape[0] > 0
        assert ds["dosage"].shape[1] == len(ds["feature_names"])
        assert ds["dataset_name"] == "DS1"

    def test_load_all_datasets(self, basic_config):
        """Test loading all datasets."""
        loader = DataLoader(basic_config)
        datasets = loader.load_all_datasets()
        assert len(datasets) == 2
        for ds in datasets:
            assert ds["dosage"].shape[0] > 0

    def test_exclude_filter(self, basic_config, tmp_data_dir):
        """Test that exclude==1 samples are removed."""
        loader = DataLoader(basic_config)
        ds = loader.load_dataset(
            str(tmp_data_dir / "ds1_dosage.csv"),
            str(tmp_data_dir / "ds1_cov.csv"),
            "DS1",
        )
        # Excluded samples should not be present
        cov = ds["covariates"]
        if "exclude" in cov.columns:
            assert (cov["exclude"] == 1).sum() == 0

    def test_sex_encoding(self, basic_config, tmp_data_dir):
        """Test that sex is encoded as M=1, F=0."""
        loader = DataLoader(basic_config)
        ds = loader.load_dataset(
            str(tmp_data_dir / "ds1_dosage.csv"),
            str(tmp_data_dir / "ds1_cov.csv"),
            "DS1",
        )
        sex = ds["covariates"]["sex"]
        valid_sex = sex.dropna()
        assert set(valid_sex.unique()).issubset({0, 1})

    def test_grade_encoding(self, basic_config, tmp_data_dir):
        """Test that grade is encoded as HGG=1, LGG=0."""
        loader = DataLoader(basic_config)
        ds = loader.load_dataset(
            str(tmp_data_dir / "ds1_dosage.csv"),
            str(tmp_data_dir / "ds1_cov.csv"),
            "DS1",
        )
        grade = ds["covariates"]["grade"]
        valid_grade = grade.dropna()
        assert set(valid_grade.unique()).issubset({0, 1})

    def test_missing_required_column(self, tmp_path):
        """Test error when required IID column is missing."""
        dos = pd.DataFrame({"sample_id": ["S1"], "HLA_A_01": [1.0]})
        cov = pd.DataFrame({"NOT_IID": ["S1"], "case": [1]})  # Missing IID
        dos.to_csv(tmp_path / "dos.csv", index=False)
        cov.to_csv(tmp_path / "cov.csv", index=False)

        config = AnalysisConfig(
            dosage_files=[str(tmp_path / "dos.csv")],
            covariate_files=[str(tmp_path / "cov.csv")],
            dataset_names=["test"],
            workers=1,
        )
        loader = DataLoader(config)
        with pytest.raises(ValueError, match="missing required columns"):
            loader.load_dataset(str(tmp_path / "dos.csv"),
                              str(tmp_path / "cov.csv"), "test")

    def test_get_stratum_indices_overall(self, basic_config, tmp_data_dir):
        """Test overall stratum returns all cases and controls."""
        loader = DataLoader(basic_config)
        ds = loader.load_dataset(
            str(tmp_data_dir / "ds1_dosage.csv"),
            str(tmp_data_dir / "ds1_cov.csv"),
            "DS1",
        )
        case_idx, ctrl_idx = loader.get_stratum_indices(ds["covariates"], "overall")
        assert case_idx.sum() > 0
        assert ctrl_idx.sum() > 0
        assert case_idx.sum() + ctrl_idx.sum() == len(ds["covariates"])

    def test_get_stratum_indices_idh_wt(self, basic_config, tmp_data_dir):
        """Test IDH wild-type stratum filters correctly."""
        loader = DataLoader(basic_config)
        ds = loader.load_dataset(
            str(tmp_data_dir / "ds1_dosage.csv"),
            str(tmp_data_dir / "ds1_cov.csv"),
            "DS1",
        )
        case_idx, ctrl_idx = loader.get_stratum_indices(ds["covariates"], "idh_wt")
        # IDH WT cases should have idh=0
        cov = ds["covariates"]
        if case_idx.sum() > 0:
            cases = cov.loc[case_idx]
            assert (cases["idh"] == 0).all()

    def test_get_survival_indices(self, basic_config, tmp_data_dir):
        """Test survival indices include only cases with survival data."""
        loader = DataLoader(basic_config)
        ds = loader.load_dataset(
            str(tmp_data_dir / "ds1_dosage.csv"),
            str(tmp_data_dir / "ds1_cov.csv"),
            "DS1",
        )
        surv_idx = loader.get_survival_indices(ds["covariates"], "overall")
        cov = ds["covariates"]
        # All should be cases
        assert (cov.loc[surv_idx, "case"] == 1).all()
        # All should have survival data
        assert cov.loc[surv_idx, "survdays"].notna().all()

    def test_prepare_covariate_matrix_full(self, basic_config, tmp_data_dir):
        """Test full covariate strategy."""
        loader = DataLoader(basic_config)
        ds = loader.load_dataset(
            str(tmp_data_dir / "ds1_dosage.csv"),
            str(tmp_data_dir / "ds1_cov.csv"),
            "DS1",
        )
        mask = np.ones(len(ds["covariates"]), dtype=bool)
        X_cov, used, final_mask = loader.prepare_covariate_matrix(
            ds["covariates"], mask, ["age", "sex"], "full"
        )
        assert X_cov is not None
        assert X_cov.shape[1] == 2
        assert "age" in used
        assert "sex" in used

    def test_prepare_covariate_matrix_reduced(self, tmp_path):
        """Test reduced strategy drops high-missingness covariates."""
        n = 100
        cov = pd.DataFrame({
            "sample_id": [f"S{i}" for i in range(n)],
            "age": np.random.normal(50, 10, n),
            "high_miss": np.where(np.random.rand(n) < 0.5, np.nan, 1),  # 50% missing
        })
        config = AnalysisConfig(
            dosage_files=[], covariate_files=[], workers=1,
            missingness_threshold=0.3,
        )
        loader = DataLoader(config)
        mask = np.ones(n, dtype=bool)
        X_cov, used, final_mask = loader.prepare_covariate_matrix(
            cov, mask, ["age", "high_miss"], "reduced"
        )
        # high_miss should be dropped (50% > 30%)
        assert "high_miss" not in used
        assert "age" in used


class TestFindCommonFeatures:
    """Tests for find_common_features."""

    def test_common_features(self):
        ds1 = {"feature_names": ["HLA_A_01", "HLA_B_01", "HLA_C_01"]}
        ds2 = {"feature_names": ["HLA_A_01", "HLA_B_01", "HLA_D_01"]}
        common = find_common_features([ds1, ds2])
        assert set(common) == {"HLA_A_01", "HLA_B_01"}

    def test_empty_datasets(self):
        assert find_common_features([]) == []

    def test_single_dataset(self):
        ds1 = {"feature_names": ["HLA_A_01", "HLA_B_01"]}
        common = find_common_features([ds1])
        assert common == ["HLA_A_01", "HLA_B_01"]


class TestComputeMissingness:
    """Tests for the compute_missingness method."""

    def test_no_missingness(self):
        """Test with no missing values."""
        cov = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3"],
            "age": [50, 60, 70],
            "sex": [1, 0, 1],
        })
        config = AnalysisConfig(workers=1)
        loader = DataLoader(config)
        mask = np.ones(3, dtype=bool)
        miss = loader.compute_missingness(cov, mask, ["age", "sex"])
        assert miss["age"] == 0.0
        assert miss["sex"] == 0.0

    def test_partial_missingness(self):
        """Test with some missing values."""
        cov = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3", "S4"],
            "age": [50, np.nan, 70, np.nan],
            "sex": [1, 0, 1, 0],
        })
        config = AnalysisConfig(workers=1)
        loader = DataLoader(config)
        mask = np.ones(4, dtype=bool)
        miss = loader.compute_missingness(cov, mask, ["age", "sex"])
        assert miss["age"] == pytest.approx(0.5)
        assert miss["sex"] == 0.0

    def test_respects_mask(self):
        """Test that only masked samples are considered."""
        cov = pd.DataFrame({
            "sample_id": ["S1", "S2", "S3", "S4"],
            "age": [50, np.nan, 70, 80],
            "sex": [1, 0, 1, 0],
        })
        config = AnalysisConfig(workers=1)
        loader = DataLoader(config)
        mask = np.array([True, False, True, True])  # Exclude S2
        miss = loader.compute_missingness(cov, mask, ["age"])
        assert miss["age"] == 0.0  # S2 is excluded, no missing in remaining

    def test_missing_column(self):
        """Test with a covariate column not present in data."""
        cov = pd.DataFrame({
            "sample_id": ["S1", "S2"],
            "age": [50, 60],
        })
        config = AnalysisConfig(workers=1)
        loader = DataLoader(config)
        mask = np.ones(2, dtype=bool)
        miss = loader.compute_missingness(cov, mask, ["age", "nonexistent"])
        assert miss["age"] == 0.0
        assert miss["nonexistent"] == 1.0  # Fully missing (not present)

