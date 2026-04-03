"""
Shared pytest fixtures and synthetic data generators for HLA analysis tests.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from hla_analysis.config import AnalysisConfig


def generate_hla_features(n_features: int = 20, seed: int = 42):
    """Generate realistic HLA feature names.

    Returns
    -------
    list of str
        Mix of classical 2-digit, 4-digit, and amino-acid features.
    """
    rng = np.random.RandomState(seed)
    genes = ["A", "B", "C", "DRB1", "DQB1", "DPB1"]
    features = []

    # Classical 2-digit
    for g in genes[:3]:
        for allele in range(1, 4):
            features.append(f"HLA_{g}_{allele:02d}")

    # Classical 4-digit
    for g in genes[:3]:
        for allele in range(1, 3):
            for sub in range(1, 3):
                features.append(f"HLA_{g}_{allele:02d}:{sub:02d}")

    # Amino acid
    for g in genes[:4]:
        for pos in [9, 11, 13]:
            for aa in ["V", "L", "D"]:
                features.append(f"AA_{g}_{pos}_{aa}")

    return features[:n_features]


def generate_synthetic_cohort(
    n_samples: int = 200,
    n_features: int = 20,
    case_frac: float = 0.5,
    effect_features: int = 3,
    effect_size: float = 0.5,
    survival_effect: bool = True,
    seed: int = 42,
):
    """Generate a synthetic cohort with HLA dosage and covariates.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of HLA features.
    case_frac : float
        Fraction of cases.
    effect_features : int
        Number of features with true effects.
    effect_size : float
        Log-OR or log-HR for effect features.
    survival_effect : bool
        Whether to generate survival data with effects.
    seed : int
        Random seed.

    Returns
    -------
    dosage_df : pd.DataFrame
        Dosage data with sample_id + feature columns.
    covariate_df : pd.DataFrame
        Covariate data with IID and all required columns.
    """
    rng = np.random.RandomState(seed)

    n_cases = int(n_samples * case_frac)
    n_controls = n_samples - n_cases

    # Generate feature names
    feature_names = generate_hla_features(n_features, seed)

    # Sample IDs
    sample_ids = [f"SAMPLE_{i:04d}" for i in range(n_samples)]

    # Generate dosage matrix (0, 1, 2) with realistic MAF
    mafs = rng.uniform(0.05, 0.45, n_features)
    dosage = np.zeros((n_samples, n_features), dtype=np.float32)
    for j in range(n_features):
        p = mafs[j]
        dosage[:, j] = rng.choice([0, 1, 2], size=n_samples,
                                   p=[(1-p)**2, 2*p*(1-p), p**2])

    # Generate case/control with effect features
    case = np.array([1] * n_cases + [0] * n_controls)
    rng.shuffle(case)

    # Add signal to effect features (higher dosage in cases)
    for j in range(min(effect_features, n_features)):
        noise = rng.normal(0, 0.1, n_samples)
        dosage[:, j] = dosage[:, j] + case * effect_size * 0.5 + noise
        dosage[:, j] = np.clip(dosage[:, j], 0, 2)

    # Covariates
    age = rng.normal(55, 15, n_samples).clip(18, 90)
    sex = rng.choice(["M", "F"], n_samples)
    pcs = rng.normal(0, 1, (n_samples, 8))

    # IDH and pq status (only for cases)
    idh = np.full(n_samples, np.nan)
    pq = np.full(n_samples, np.nan)
    grade = np.full(n_samples, np.nan, dtype=object)
    treated = np.full(n_samples, np.nan)

    case_idx = np.where(case == 1)[0]
    idh[case_idx] = rng.choice([0, 1], len(case_idx), p=[0.4, 0.6])
    for i in case_idx:
        if idh[i] == 1:
            pq[i] = rng.choice([0, 1], p=[0.6, 0.4])
        else:
            pq[i] = 0
    grade_arr = np.where(idh == 0, "HGG", "LGG")
    grade[case_idx] = grade_arr[case_idx]
    treated[case_idx] = rng.choice([0, 1], len(case_idx), p=[0.3, 0.7])

    # Survival data (cases only)
    survdays = np.full(n_samples, np.nan)
    vstatus = np.full(n_samples, np.nan)

    for i in case_idx:
        baseline_hazard = 0.001
        if survival_effect and n_features > 0:
            risk = np.exp(dosage[i, 0] * effect_size * 0.3)
        else:
            risk = 1.0
        survdays[i] = rng.exponential(1000 / (baseline_hazard * risk))
        vstatus[i] = rng.choice([0, 1], p=[0.3, 0.7])

    # Exclude column (mark a few for exclusion)
    exclude = np.zeros(n_samples)
    exclude_idx = rng.choice(n_samples, size=max(1, n_samples // 50), replace=False)
    exclude[exclude_idx] = 1

    # Build DataFrames
    dosage_df = pd.DataFrame(dosage, columns=feature_names)
    dosage_df.insert(0, "sample_id", sample_ids)

    covariate_df = pd.DataFrame({
        "IID": sample_ids,
        "dataset": "synthetic",
        "age": age,
        "sex": sex,
        "case": case,
        "grade": grade,
        "idh": idh,
        "pq": pq,
        "treated": treated,
        "survdays": survdays,
        "vstatus": vstatus,
        "exclude": exclude,
    })
    for k in range(8):
        covariate_df[f"PC{k+1}"] = pcs[:, k]

    return dosage_df, covariate_df


@pytest.fixture
def synthetic_small():
    """Small synthetic cohort (50 samples, 15 features)."""
    return generate_synthetic_cohort(n_samples=50, n_features=15, seed=42)


@pytest.fixture
def synthetic_medium():
    """Medium synthetic cohort (200 samples, 20 features)."""
    return generate_synthetic_cohort(n_samples=200, n_features=20, seed=123)


@pytest.fixture
def synthetic_two_datasets():
    """Two synthetic datasets for meta-analysis testing."""
    ds1 = generate_synthetic_cohort(n_samples=100, n_features=15, seed=42,
                                     effect_size=0.5)
    ds2 = generate_synthetic_cohort(n_samples=100, n_features=15, seed=99,
                                     effect_size=0.5)
    return ds1, ds2


@pytest.fixture
def tmp_data_dir(synthetic_two_datasets, tmp_path):
    """Write two synthetic datasets to temp CSV files."""
    (dos1, cov1), (dos2, cov2) = synthetic_two_datasets

    dos1.to_csv(tmp_path / "ds1_dosage.csv", index=False)
    cov1.to_csv(tmp_path / "ds1_cov.csv", index=False)
    dos2.to_csv(tmp_path / "ds2_dosage.csv", index=False)
    cov2.to_csv(tmp_path / "ds2_cov.csv", index=False)

    return tmp_path


@pytest.fixture
def basic_config(tmp_data_dir):
    """Basic AnalysisConfig pointing to two synthetic datasets."""
    return AnalysisConfig(
        dosage_files=[
            str(tmp_data_dir / "ds1_dosage.csv"),
            str(tmp_data_dir / "ds2_dosage.csv"),
        ],
        covariate_files=[
            str(tmp_data_dir / "ds1_cov.csv"),
            str(tmp_data_dir / "ds2_cov.csv"),
        ],
        dataset_names=["DS1", "DS2"],
        output_dir=str(tmp_data_dir / "results"),
        workers=1,  # Serial for testing
        chunk_size=10,
        strata=["overall"],
        covariate_strategies=["full"],
        feature_types=["classical_2digit", "classical_4digit", "amino_acid"],
        plots=[],
        log_level="WARNING",
    )


@pytest.fixture
def synthetic_with_missingness():
    """Synthetic cohort with controlled missingness in covariates."""
    rng = np.random.RandomState(55)
    n = 150
    n_features = 10

    # Base cohort
    dos_df, cov_df = generate_synthetic_cohort(
        n_samples=n, n_features=n_features, seed=55,
    )

    # Inject controlled missingness
    # age: 20% missing
    age_miss = rng.choice(n, size=int(n * 0.20), replace=False)
    cov_df.loc[cov_df.index[age_miss], "age"] = np.nan

    # grade: 40% missing among cases
    case_rows = cov_df[cov_df["case"] == 1].index
    grade_miss = rng.choice(case_rows, size=int(len(case_rows) * 0.40), replace=False)
    cov_df.loc[grade_miss, "grade"] = np.nan

    return dos_df, cov_df


@pytest.fixture
def sensitivity_config(synthetic_two_datasets, tmp_path):
    """AnalysisConfig with sensitivity_analysis enabled, pointing to two datasets."""
    (dos1, cov1), (dos2, cov2) = synthetic_two_datasets

    # Inject missingness in age for dataset 1
    rng = np.random.RandomState(77)
    miss_idx = rng.choice(len(cov1), size=int(len(cov1) * 0.15), replace=False)
    cov1.iloc[miss_idx, cov1.columns.get_loc("age")] = np.nan

    dos1.to_csv(tmp_path / "ds1_dosage.csv", index=False)
    cov1.to_csv(tmp_path / "ds1_cov.csv", index=False)
    dos2.to_csv(tmp_path / "ds2_dosage.csv", index=False)
    cov2.to_csv(tmp_path / "ds2_cov.csv", index=False)

    return AnalysisConfig(
        dosage_files=[
            str(tmp_path / "ds1_dosage.csv"),
            str(tmp_path / "ds2_dosage.csv"),
        ],
        covariate_files=[
            str(tmp_path / "ds1_cov.csv"),
            str(tmp_path / "ds2_cov.csv"),
        ],
        dataset_names=["DS1", "DS2"],
        output_dir=str(tmp_path / "results_sens"),
        workers=1,
        chunk_size=10,
        strata=["overall"],
        sensitivity_analysis=True,
        feature_types=["classical_2digit", "classical_4digit", "amino_acid"],
        plots=[],
        log_level="WARNING",
        min_carriers=3,
        min_events=2,
    )

