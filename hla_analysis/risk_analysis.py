"""
Logistic regression risk analysis for HLA allele associations.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from hla_analysis.config import AnalysisConfig
from hla_analysis.utils import benjamini_hochberg, safe_exp, results_to_dataframe

logger = logging.getLogger(__name__)


def _fit_logistic_single(
    dosage_col: np.ndarray,
    y: np.ndarray,
    X_cov: Optional[np.ndarray],
    feature_name: str,
    min_carriers: int,
) -> Optional[Dict[str, Any]]:
    """Fit a single logistic regression model.

    Parameters
    ----------
    dosage_col : np.ndarray
        Dosage values for one HLA feature (n_samples,).
    y : np.ndarray
        Case/control labels (n_samples,).
    X_cov : np.ndarray or None
        Covariate matrix (n_samples, n_covariates).
    feature_name : str
        Name of the feature.
    min_carriers : int
        Minimum carriers in each group.

    Returns
    -------
    dict or None
        Result dictionary with beta, SE, OR, CI, p-value, etc. None if skipped.
    """
    import statsmodels.api as sm

    # Check carrier count in each group
    cases_mask = y == 1
    carriers_cases = (dosage_col[cases_mask] > 0).sum()
    carriers_controls = (dosage_col[~cases_mask] > 0).sum()

    if carriers_cases < min_carriers or carriers_controls < min_carriers:
        return None

    # Check for zero-variance feature
    if np.std(dosage_col) < 1e-10:
        return None

    # Build design matrix
    if X_cov is not None:
        X = np.column_stack([dosage_col, X_cov])
    else:
        X = dosage_col.reshape(-1, 1)

    X = sm.add_constant(X, has_constant="add")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(y, X)
            result = model.fit(disp=0, maxiter=100, method="newton")

        if not result.mle_retvals.get("converged", True):
            return {
                "feature": feature_name,
                "converged": False,
                "beta": np.nan,
                "se": np.nan,
                "or_val": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "pvalue": np.nan,
                "n_cases": int(cases_mask.sum()),
                "n_controls": int((~cases_mask).sum()),
                "carriers_cases": int(carriers_cases),
                "carriers_controls": int(carriers_controls),
            }

        # Extract HLA feature coefficient (index 1, after constant)
        beta = result.params[1]
        se = result.bse[1]
        or_val = safe_exp(beta)
        ci_lower = safe_exp(beta - 1.96 * se)
        ci_upper = safe_exp(beta + 1.96 * se)
        pvalue = result.pvalues[1]

        return {
            "feature": feature_name,
            "converged": True,
            "beta": float(beta),
            "se": float(se),
            "or_val": float(or_val),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "pvalue": float(pvalue),
            "n_cases": int(cases_mask.sum()),
            "n_controls": int((~cases_mask).sum()),
            "carriers_cases": int(carriers_cases),
            "carriers_controls": int(carriers_controls),
        }

    except Exception as e:
        logger.debug("Logistic regression failed for %s: %s", feature_name, e)
        return {
            "feature": feature_name,
            "converged": False,
            "beta": np.nan,
            "se": np.nan,
            "or_val": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "pvalue": np.nan,
            "n_cases": int(cases_mask.sum()),
            "n_controls": int((~cases_mask).sum()),
            "carriers_cases": int(carriers_cases),
            "carriers_controls": int(carriers_controls),
        }


def _process_feature_chunk_risk(
    chunk_indices: List[int],
    dosage_matrix: np.ndarray,
    y: np.ndarray,
    X_cov: Optional[np.ndarray],
    feature_names: List[str],
    min_carriers: int,
) -> List[Dict[str, Any]]:
    """Process a chunk of features for risk analysis.

    Parameters
    ----------
    chunk_indices : list of int
        Indices into dosage_matrix columns and feature_names.
    dosage_matrix : np.ndarray
        Full dosage matrix (n_samples, n_features).
    y : np.ndarray
        Outcome vector.
    X_cov : np.ndarray or None
        Covariate matrix.
    feature_names : list of str
        Feature names.
    min_carriers : int
        Minimum carrier threshold.

    Returns
    -------
    list of dict
        Results for each feature in the chunk.
    """
    results = []
    for idx in chunk_indices:
        res = _fit_logistic_single(
            dosage_matrix[:, idx], y, X_cov, feature_names[idx], min_carriers
        )
        if res is not None:
            results.append(res)
    return results


class RiskAnalyzer:
    """Perform logistic regression risk analysis across features.

    Parameters
    ----------
    config : AnalysisConfig
        Pipeline configuration.
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def analyze_stratum(
        self,
        dosage_matrix: np.ndarray,
        y: np.ndarray,
        X_cov: Optional[np.ndarray],
        feature_names: List[str],
        dataset_name: str,
        stratum: str,
        strategy: str,
        covariate_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run risk analysis for one dataset-stratum-strategy combination.

        Parameters
        ----------
        dosage_matrix : np.ndarray
            Dosage values (n_samples, n_features), float32.
        y : np.ndarray
            Case labels (n_samples,).
        X_cov : np.ndarray or None
            Covariate matrix (n_samples, n_covariates).
        feature_names : list of str
            Feature names corresponding to dosage columns.
        dataset_name : str
            Name of the dataset.
        stratum : str
            Name of the stratum.
        strategy : str
            Covariate strategy name.
        covariate_names : list of str, optional
            Names of covariates used.

        Returns
        -------
        pd.DataFrame
            Results with columns: feature, beta, se, or_val, ci_lower, ci_upper,
            pvalue, fdr, n_cases, n_controls, converged, dataset, stratum, strategy.
        """
        n_features = len(feature_names)
        logger.info(
            "Risk analysis: dataset=%s, stratum=%s, strategy=%s, "
            "n_samples=%d, n_features=%d",
            dataset_name, stratum, strategy, dosage_matrix.shape[0], n_features,
        )

        # Create chunks
        chunk_size = self.config.chunk_size
        chunks = [
            list(range(i, min(i + chunk_size, n_features)))
            for i in range(0, n_features, chunk_size)
        ]

        # Process chunks
        all_results = []
        workers = min(self.config.workers, len(chunks))

        if workers > 1 and len(chunks) > 1:
            func = partial(
                _process_feature_chunk_risk,
                dosage_matrix=dosage_matrix,
                y=y,
                X_cov=X_cov,
                feature_names=feature_names,
                min_carriers=self.config.min_carriers,
            )
            try:
                with Pool(processes=workers) as pool:
                    chunk_results = pool.map(func, chunks)
                for cr in chunk_results:
                    all_results.extend(cr)
            except Exception as e:
                logger.warning("Parallel processing failed, falling back to serial: %s", e)
                for chunk in chunks:
                    all_results.extend(
                        _process_feature_chunk_risk(
                            chunk, dosage_matrix, y, X_cov, feature_names,
                            self.config.min_carriers,
                        )
                    )
        else:
            for chunk in chunks:
                all_results.extend(
                    _process_feature_chunk_risk(
                        chunk, dosage_matrix, y, X_cov, feature_names,
                        self.config.min_carriers,
                    )
                )

        if not all_results:
            logger.warning("No features passed filters for dataset=%s, stratum=%s, strategy=%s",
                           dataset_name, stratum, strategy)
            return pd.DataFrame()

        df = results_to_dataframe(all_results)
        df["dataset"] = dataset_name
        df["stratum"] = stratum
        df["strategy"] = strategy
        if covariate_names:
            df["covariates_used"] = ",".join(covariate_names)

        # FDR correction
        valid_p = df["pvalue"].values
        df["fdr"] = benjamini_hochberg(valid_p, self.config.fdr_threshold)

        n_sig = (df["fdr"] < self.config.fdr_threshold).sum()
        logger.info(
            "Risk results: dataset=%s, stratum=%s, strategy=%s: "
            "%d features tested, %d significant (FDR < %.2f)",
            dataset_name, stratum, strategy, len(df), n_sig, self.config.fdr_threshold,
        )

        return df

    def run_single_model(
        self,
        dosage_col: np.ndarray,
        y: np.ndarray,
        X_cov: Optional[np.ndarray],
        feature_name: str = "feature",
    ) -> Optional[Dict[str, Any]]:
        """Fit a single logistic regression model (convenience method).

        Parameters
        ----------
        dosage_col : np.ndarray
            Dosage values for one feature.
        y : np.ndarray
            Case labels.
        X_cov : np.ndarray or None
            Covariate matrix.
        feature_name : str
            Feature name.

        Returns
        -------
        dict or None
        """
        return _fit_logistic_single(
            dosage_col, y, X_cov, feature_name, self.config.min_carriers
        )
