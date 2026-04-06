"""
Data loading, merging, validation, and stratification.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hla_analysis.config import AnalysisConfig, STRATA_DEFINITIONS
from hla_analysis.utils import encode_sex, encode_grade, classify_features
from hla_analysis.vcf_parser import detect_dosage_format, parse_vcf_to_dosage_df

logger = logging.getLogger(__name__)


def compute_maf(dosage_col: np.ndarray) -> float:
    """Compute minor allele frequency from a dosage column.

    Parameters
    ----------
    dosage_col : np.ndarray
        Dosage values for one feature (0-2 scale).

    Returns
    -------
    float
        Minor allele frequency: mean(dosage) / 2.
    """
    valid = dosage_col[~np.isnan(dosage_col)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid) / 2.0)


class DataLoader:
    """Load, validate, and prepare HLA dosage + covariate data.

    Parameters
    ----------
    config : AnalysisConfig
        Pipeline configuration.
    """

    # Required covariate columns
    REQUIRED_COV_COLS = {"IID"}
    # Expected covariate columns (may not all be present)
    EXPECTED_COV_COLS = {
        "IID", "dataset", "age", "sex", "case", "grade",
        "idh", "pq", "treated", "survdays", "vstatus", "exclude",
        "rad", "chemo",
    }

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def load_dataset(self, dosage_path: str, covariate_path: str,
                     dataset_name: str) -> Dict:
        """Load and merge one dataset.

        Parameters
        ----------
        dosage_path : str
            Path to HLA dosage CSV.
        covariate_path : str
            Path to covariate CSV.
        dataset_name : str
            Name for this dataset.

        Returns
        -------
        dict with keys:
            'covariates': pd.DataFrame (covariate data, encoded)
            'dosage': np.ndarray (float32, samples x features)
            'feature_names': list of str
            'sample_ids': list of str
            'dataset_name': str
        """
        logger.info("Loading dataset %s: dosage=%s, cov=%s",
                     dataset_name, dosage_path, covariate_path)

        # Load covariate data
        cov = pd.read_csv(covariate_path)
        self._validate_covariates(cov, dataset_name)

        # Load dosage data — auto-detect CSV vs VCF
        dosage_df, r2_info = self._load_dosage(dosage_path, dataset_name)
        if "sample_id" not in dosage_df.columns:
            raise ValueError(
                f"Dosage file {dosage_path} must have a 'sample_id' column"
            )

        # Identify HLA feature columns
        feature_cols = [c for c in dosage_df.columns if c != "sample_id"]
        logger.info("Dataset %s: %d features in dosage file", dataset_name, len(feature_cols))

        # ── R² filtering ──
        if r2_info and self.config.min_imputation_r2 > 0:
            before_r2 = len(feature_cols)
            feature_cols = [
                f for f in feature_cols
                if r2_info.get(f, 1.0) >= self.config.min_imputation_r2
            ]
            n_dropped_r2 = before_r2 - len(feature_cols)
            if n_dropped_r2 > 0:
                logger.info(
                    "Dataset %s: dropped %d features with R² < %.2f (%d remain)",
                    dataset_name, n_dropped_r2, self.config.min_imputation_r2,
                    len(feature_cols),
                )

        # Filter feature types
        classified = classify_features(feature_cols)
        selected_features = []
        for ftype in self.config.feature_types:
            selected_features.extend(classified.get(ftype, []))

        if not selected_features:
            logger.warning("Dataset %s: no features match selected types %s",
                           dataset_name, self.config.feature_types)

        logger.info("Dataset %s: %d features selected after type filtering",
                     dataset_name, len(selected_features))

        # ── MAF filtering ──
        before_maf = len(selected_features)
        maf_passed = []
        for feat in selected_features:
            dosage_vals = dosage_df[feat].values
            maf = compute_maf(dosage_vals)
            # Apply different thresholds for alleles vs amino acid positions
            if feat.startswith("AA_"):
                threshold = self.config.maf_threshold_aa
            else:
                threshold = self.config.maf_threshold_allele
            if maf >= threshold:
                maf_passed.append(feat)
        selected_features = maf_passed
        n_dropped_maf = before_maf - len(selected_features)
        if n_dropped_maf > 0:
            logger.info(
                "Dataset %s: dropped %d features below MAF threshold (%d remain)",
                dataset_name, n_dropped_maf, len(selected_features),
            )

        # Merge on IID / sample_id
        cov = cov.rename(columns={"IID": "sample_id"}) if "IID" in cov.columns else cov
        merged = dosage_df[["sample_id"] + selected_features].merge(
            cov, on="sample_id", how="inner"
        )

        n_before = len(dosage_df)
        n_after = len(merged)
        if n_after < n_before:
            logger.info("Dataset %s: %d/%d samples matched after merge",
                         dataset_name, n_after, n_before)

        # Apply exclusion filter
        if "exclude" in merged.columns:
            n_excluded = (merged["exclude"] == 1).sum()
            if n_excluded > 0:
                merged = merged[merged["exclude"] != 1].copy()
                logger.info("Dataset %s: excluded %d samples (exclude==1)",
                             dataset_name, n_excluded)

        # Encode variables
        if "sex" in merged.columns:
            merged["sex"] = encode_sex(merged["sex"])
        if "grade" in merged.columns:
            merged["grade"] = encode_grade(merged["grade"])

        # Ensure numeric types for key columns
        for col in ["case", "idh", "pq", "treated", "age", "survdays", "vstatus",
                     "rad", "chemo"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

        # Extract dosage matrix as float32
        dosage_matrix = merged[selected_features].values.astype(np.float32)

        # Covariate columns (everything except features and sample_id)
        cov_cols = [c for c in merged.columns if c not in selected_features and c != "sample_id"]
        covariate_df = merged[["sample_id"] + cov_cols].copy()

        logger.info(
            "Dataset %s: %d samples, %d features, %d covariate columns",
            dataset_name, len(merged), len(selected_features), len(cov_cols),
        )

        return {
            "covariates": covariate_df,
            "dosage": dosage_matrix,
            "feature_names": selected_features,
            "sample_ids": merged["sample_id"].tolist(),
            "dataset_name": dataset_name,
        }

    def _validate_covariates(self, cov: pd.DataFrame, dataset_name: str):
        """Validate covariate DataFrame."""
        missing_required = self.REQUIRED_COV_COLS - set(cov.columns)
        if missing_required:
            raise ValueError(
                f"Dataset {dataset_name}: covariate file missing required columns: {missing_required}"
            )
        if "case" not in cov.columns:
            logger.warning("Dataset %s: no 'case' column found in covariates", dataset_name)

    def _load_dosage(self, dosage_path: str, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Load a dosage file, dispatching to CSV or VCF parser.

        Parameters
        ----------
        dosage_path : str
            Path to dosage file (CSV or VCF/VCF.GZ).
        dataset_name : str
            Name for logging.

        Returns
        -------
        tuple of (pd.DataFrame, dict)
            Dosage data with ``sample_id`` column + feature columns,
            and a dict mapping feature names to R² values (empty if CSV).
        """
        fmt = self.config.dosage_format
        if fmt == "auto":
            fmt = detect_dosage_format(dosage_path)

        r2_info: Dict[str, float] = {}

        if fmt == "vcf":
            logger.info("Dataset %s: loading VCF dosage from %s", dataset_name, dosage_path)
            df = parse_vcf_to_dosage_df(
                dosage_path,
                field=self.config.vcf_field,
                filter_prefixes=self.config.vcf_filter_prefixes,
                include_snps=self.config.include_snps,
                normalize_ids=True,
            )
            # Also parse R² from VCF INFO field
            r2_info = self._parse_vcf_r2(dosage_path, df.columns.tolist())
            return df, r2_info
        else:
            logger.info("Dataset %s: loading CSV dosage from %s", dataset_name, dosage_path)
            return pd.read_csv(dosage_path), r2_info

    def _parse_vcf_r2(self, vcf_path: str, feature_names: List[str]) -> Dict[str, float]:
        """Parse R² values from VCF INFO field for quality filtering.

        Parameters
        ----------
        vcf_path : str
            Path to VCF file.
        feature_names : list of str
            Feature names (after normalization) to look for.

        Returns
        -------
        dict of str -> float
            Mapping from feature name to R² value.
        """
        import gzip

        r2_info: Dict[str, float] = {}
        feature_set = set(feature_names) - {"sample_id"}

        opener = gzip.open if vcf_path.endswith(".gz") else open
        open_kwargs = {"mode": "rt", "encoding": "utf-8"} if vcf_path.endswith(".gz") else {"mode": "r", "encoding": "utf-8"}

        try:
            with opener(vcf_path, **open_kwargs) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    cols = line.rstrip("\n\r").split("\t", 9)
                    if len(cols) < 8:
                        continue
                    variant_id = cols[2]
                    # Normalize to match our convention
                    norm_id = variant_id.replace("*", "_")
                    if norm_id not in feature_set:
                        continue
                    # Parse INFO field for R2
                    info = cols[7]
                    r2_val = None
                    for kv in info.split(";"):
                        if kv.startswith("R2=") or kv.startswith("DR2="):
                            try:
                                r2_val = float(kv.split("=", 1)[1])
                            except (ValueError, IndexError):
                                pass
                            break
                    if r2_val is not None:
                        r2_info[norm_id] = r2_val
        except Exception as e:
            logger.warning("Failed to parse R² from VCF %s: %s", vcf_path, e)

        logger.info("Parsed R² for %d features from VCF", len(r2_info))
        return r2_info

    def load_all_datasets(self) -> List[Dict]:
        """Load all datasets specified in config.

        Returns
        -------
        list of dict
            One entry per dataset, as returned by :meth:`load_dataset`.
        """
        datasets = []
        for i, (dos_path, cov_path) in enumerate(
            zip(self.config.dosage_files, self.config.covariate_files)
        ):
            name = (
                self.config.dataset_names[i]
                if self.config.dataset_names
                else f"dataset_{i}"
            )
            ds = self.load_dataset(dos_path, cov_path, name)
            datasets.append(ds)
        return datasets

    def get_stratum_indices(self, covariates: pd.DataFrame,
                            stratum: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get case and control sample indices for a given stratum.

        For risk analysis: cases are filtered by stratum criteria, all controls are used.
        Controls never have IDH/pq status — they are shared across strata.

        Parameters
        ----------
        covariates : pd.DataFrame
            Covariate DataFrame with 'case' column.
        stratum : str
            Name of the stratum.

        Returns
        -------
        case_idx : np.ndarray
            Boolean index for cases.
        control_idx : np.ndarray
            Boolean index for controls.
        """
        if "case" not in covariates.columns:
            raise ValueError("Covariate data must have a 'case' column for stratification.")

        control_idx = (covariates["case"] == 0).values
        case_base = (covariates["case"] == 1).values

        stratum_def = STRATA_DEFINITIONS.get(stratum)
        if stratum_def is None:
            raise ValueError(f"Unknown stratum: {stratum}")

        case_filter = stratum_def["case_filter"]
        if case_filter is None:
            # overall: all cases
            case_idx = case_base
        else:
            # Apply additional filters on cases
            case_idx = case_base.copy()
            for col, val in case_filter.items():
                if col in covariates.columns:
                    col_vals = covariates[col].values
                    case_idx = case_idx & (col_vals == val)
                else:
                    logger.warning("Stratum %s requires column '%s' which is missing; "
                                   "skipping this filter", stratum, col)

        n_cases = case_idx.sum()
        n_controls = control_idx.sum()
        logger.debug("Stratum %s: %d cases, %d controls", stratum, n_cases, n_controls)

        return case_idx, control_idx

    def get_survival_indices(self, covariates: pd.DataFrame,
                              stratum: str) -> np.ndarray:
        """Get sample indices for survival analysis (cases only).

        Parameters
        ----------
        covariates : pd.DataFrame
            Covariate DataFrame.
        stratum : str
            Name of the stratum.

        Returns
        -------
        np.ndarray
            Boolean index for samples to include.
        """
        case_idx, _ = self.get_stratum_indices(covariates, stratum)

        # Further require non-missing survival data
        has_surv = True
        if "survdays" in covariates.columns:
            has_surv = has_surv & covariates["survdays"].notna().values
        else:
            logger.warning("No 'survdays' column for survival analysis")
            return np.zeros(len(covariates), dtype=bool)

        if "vstatus" in covariates.columns:
            has_surv = has_surv & covariates["vstatus"].notna().values

        return case_idx & has_surv


    def compute_missingness(
        self,
        covariates: pd.DataFrame,
        sample_mask: np.ndarray,
        covariate_names: List[str],
    ) -> Dict[str, float]:
        """Compute per-covariate missingness for a sample subset.

        Parameters
        ----------
        covariates : pd.DataFrame
            Full covariate DataFrame.
        sample_mask : np.ndarray
            Boolean mask selecting relevant samples.
        covariate_names : list of str
            Covariate column names to assess.

        Returns
        -------
        dict of str -> float
            Mapping from covariate name to fraction of missing values
            (0.0 = no missingness, 1.0 = all missing).
        """
        subset = covariates.loc[sample_mask]
        result: Dict[str, float] = {}
        for cov in covariate_names:
            if cov in subset.columns:
                miss_frac = float(subset[cov].isna().mean())
            else:
                miss_frac = 1.0  # column not present => fully missing
            result[cov] = miss_frac
        return result

    def prepare_covariate_matrix(
        self,
        covariates: pd.DataFrame,
        sample_mask: np.ndarray,
        covariate_names: List[str],
        strategy: str = "full",
    ) -> Tuple[Optional[np.ndarray], List[str], np.ndarray]:
        """Build the covariate design matrix for a sample subset.

        Automatically drops covariates that are all-NaN or have zero variance
        in the current subset to prevent model failures (e.g. rad/chemo in TCGA,
        treated being constant).

        Parameters
        ----------
        covariates : pd.DataFrame
            Full covariate DataFrame.
        sample_mask : np.ndarray
            Boolean mask for samples to include.
        covariate_names : list of str
            Requested covariate column names.
        strategy : str
            'full' or 'reduced'.

        Returns
        -------
        X_cov : np.ndarray or None
            Covariate matrix (n_samples, n_covariates). None if no covariates remain.
        used_covariates : list of str
            Names of covariates actually used.
        final_mask : np.ndarray
            Updated boolean mask after dropping missing rows.
        """
        subset = covariates.loc[sample_mask].copy()

        # Filter to available covariates
        available = [c for c in covariate_names if c in subset.columns]
        if not available:
            logger.warning("No requested covariates available; running without covariates")
            return None, [], sample_mask

        # ── Drop covariates that are all-NaN in this subset ──
        kept = []
        for c in available:
            if subset[c].isna().all():
                logger.warning(
                    "Covariate '%s' is ALL NaN for the current sample subset — "
                    "auto-dropping from model", c
                )
            else:
                kept.append(c)
        available = kept

        if not available:
            logger.warning("All covariates are all-NaN; running without covariates")
            return None, [], sample_mask

        if strategy == "reduced":
            # Drop covariates with too much missingness
            kept = []
            for c in available:
                miss_frac = subset[c].isna().mean()
                if miss_frac <= self.config.missingness_threshold:
                    kept.append(c)
                else:
                    logger.info("Reduced strategy: dropping covariate '%s' "
                                "(%.1f%% missing > %.0f%% threshold)",
                                c, miss_frac * 100, self.config.missingness_threshold * 100)
            available = kept

        if not available:
            logger.warning("All covariates dropped in reduced strategy; running without covariates")
            return None, [], sample_mask

        # Drop samples with any remaining missing covariates
        cov_data = subset[available]
        complete = cov_data.notna().all(axis=1)

        n_dropped = (~complete).sum()
        if n_dropped > 0:
            logger.info("Dropped %d samples with missing covariates (%s strategy)",
                         n_dropped, strategy)

        # Build final mask: original mask positions that are also complete
        final_mask = sample_mask.copy()
        # Get indices where sample_mask is True
        mask_positions = np.where(sample_mask)[0]
        drop_positions = mask_positions[~complete.values]
        final_mask[drop_positions] = False

        X_cov = cov_data.loc[complete].values.astype(np.float64)

        # ── Drop zero-variance covariates ──
        # (e.g., 'treated' being all 1 in TCGA cases)
        variances = np.var(X_cov, axis=0)
        zero_var_mask = variances < 1e-10
        if zero_var_mask.any():
            dropped_names = [available[i] for i in range(len(available)) if zero_var_mask[i]]
            for dname in dropped_names:
                logger.warning(
                    "Covariate '%s' has zero variance in current subset — "
                    "auto-dropping from model", dname
                )
            keep_cols = ~zero_var_mask
            X_cov = X_cov[:, keep_cols]
            available = [available[i] for i in range(len(available)) if keep_cols[i]]

        if X_cov.shape[1] == 0:
            logger.warning("All covariates dropped (zero variance); running without covariates")
            return None, [], final_mask

        return X_cov, available, final_mask


def find_common_features(datasets: List[Dict]) -> List[str]:
    """Find HLA features common to all datasets.

    Parameters
    ----------
    datasets : list of dict
        Each entry from DataLoader.load_dataset().

    Returns
    -------
    list of str
        Feature names present in all datasets (preserving order of first dataset).
    """
    if not datasets:
        return []
    common = set(datasets[0]["feature_names"])
    for ds in datasets[1:]:
        common &= set(ds["feature_names"])
    # Preserve order from first dataset
    return [f for f in datasets[0]["feature_names"] if f in common]
