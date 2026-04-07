"""
Logistic regression risk analysis for HLA allele associations.
Supports Firth's penalized logistic regression to handle separation.
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


def _fit_firth_logistic(
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Optional[Dict[str, Any]]:
    """Fit Firth's penalized logistic regression.

    Implements Firth's bias-reduced logistic regression which adds
    Jeffreys invariant prior (0.5 * hat-diagonal penalty) to prevent
    infinite coefficients from complete/quasi-complete separation.

    Parameters
    ----------
    y : np.ndarray
        Binary outcome (0/1).
    X : np.ndarray
        Design matrix including intercept.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on log-likelihood change.

    Returns
    -------
    dict or None
        beta, se arrays and convergence info, or None if failed.
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)

    prev_ll = -np.inf
    converged = False

    for iteration in range(max_iter):
        # Compute probabilities
        eta = X @ beta
        eta = np.clip(eta, -500, 500)
        mu = 1.0 / (1.0 + np.exp(-eta))
        mu = np.clip(mu, 1e-10, 1.0 - 1e-10)

        # Weight matrix W = diag(mu * (1 - mu))
        W = mu * (1.0 - mu)

        # Information matrix: X^T W X
        XtW = X.T * W[np.newaxis, :]
        info_mat = XtW @ X

        # Hat matrix diagonal: h_ii = W_ii * X_i (X^T W X)^{-1} X_i^T
        try:
            info_inv = np.linalg.inv(info_mat)
        except np.linalg.LinAlgError:
            # Add small ridge for numerical stability
            try:
                info_inv = np.linalg.inv(info_mat + 1e-6 * np.eye(p))
            except np.linalg.LinAlgError:
                return None

        # Compute hat diagonal efficiently
        # H = W^{1/2} X (X^T W X)^{-1} X^T W^{1/2}
        # h_ii = W_i * X_i @ info_inv @ X_i
        hat_diag = np.einsum('ij,jk,ik->i', X * W[:, np.newaxis], info_inv, X)

        # Firth-penalized score: add 0.5 * h_i * (1 - 2*mu_i) to score
        score = X.T @ (y - mu) + 0.5 * X.T @ (hat_diag * (1.0 - 2.0 * mu))

        # Penalized log-likelihood
        ll = np.sum(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)) + 0.5 * np.log(np.linalg.det(info_mat) + 1e-300)

        if np.isnan(ll) or np.isinf(ll):
            break

        if abs(ll - prev_ll) < tol and iteration > 0:
            converged = True
            break
        prev_ll = ll

        # Newton step
        try:
            step = info_inv @ score
        except Exception:
            break

        beta = beta + step

    # Final SE computation
    eta = X @ beta
    eta = np.clip(eta, -500, 500)
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-10, 1.0 - 1e-10)
    W = mu * (1.0 - mu)
    XtW = X.T * W[np.newaxis, :]
    info_mat = XtW @ X

    try:
        var_cov = np.linalg.inv(info_mat)
        se = np.sqrt(np.maximum(np.diag(var_cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)

    return {
        "beta": beta,
        "se": se,
        "converged": converged,
    }


def _check_extreme_beta(
    result: Dict[str, Any],
    max_abs_beta: float,
) -> Dict[str, Any]:
    """Post-hoc filter: mark results with extreme beta/SE as unstable.

    Parameters
    ----------
    result : dict
        Result from logistic regression fitting.
    max_abs_beta : float
        Threshold for |beta| and SE.

    Returns
    -------
    dict
        Original result if OK, or result with NaN estimates and note.
    """
    beta = result.get("beta", np.nan)
    se = result.get("se", np.nan)
    if (not np.isnan(beta) and abs(beta) > max_abs_beta) or \
       (not np.isnan(se) and se > max_abs_beta):
        logger.debug(
            "Feature %s: extreme estimate filtered (|beta|=%.2f, SE=%.2f > %.1f)",
            result.get("feature", "?"), abs(beta) if not np.isnan(beta) else 0,
            se if not np.isnan(se) else 0, max_abs_beta,
        )
        result = dict(result)
        result["beta"] = np.nan
        result["se"] = np.nan
        result["or_val"] = np.nan
        result["ci_lower"] = np.nan
        result["ci_upper"] = np.nan
        result["pvalue"] = np.nan
        result["note"] = "filtered_extreme_beta"
    return result


def _fit_logistic_single(
    dosage_col: np.ndarray,
    y: np.ndarray,
    X_cov: Optional[np.ndarray],
    feature_name: str,
    min_carriers: int,
    use_firth: bool = False,
    max_abs_beta: float = 10.0,
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
    use_firth : bool
        If True, use Firth's penalized logistic regression when needed.

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
            # Try Firth if standard failed to converge
            if use_firth:
                firth_res = _fit_firth_logistic(y, X)
                if firth_res is not None and firth_res["converged"]:
                    beta = firth_res["beta"][1]
                    se = firth_res["se"][1]
                    or_val = safe_exp(beta)
                    ci_lower = safe_exp(beta - 1.96 * se)
                    ci_upper = safe_exp(beta + 1.96 * se)
                    z = beta / se if se > 0 else 0.0
                    pvalue = float(2 * sp_stats.norm.sf(abs(z)))

                    return _check_extreme_beta({
                        "feature": feature_name,
                        "converged": True,
                        "method": "firth",
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
                    }, max_abs_beta)

            return {
                "feature": feature_name,
                "converged": False,
                "method": "standard",
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

        # Check for quasi-complete separation indicators
        # If beta > 5 or SE > 10, refit with Firth
        if use_firth and (abs(beta) > 5 or se > 10):
            logger.debug(
                "Feature %s shows possible separation (beta=%.2f, SE=%.2f) — "
                "refitting with Firth's method",
                feature_name, beta, se,
            )
            firth_res = _fit_firth_logistic(y, X)
            if firth_res is not None and firth_res["converged"]:
                beta = firth_res["beta"][1]
                se = firth_res["se"][1]
                or_val = safe_exp(beta)
                ci_lower = safe_exp(beta - 1.96 * se)
                ci_upper = safe_exp(beta + 1.96 * se)
                z = beta / se if se > 0 else 0.0
                pvalue = float(2 * sp_stats.norm.sf(abs(z)))
                return _check_extreme_beta({
                    "feature": feature_name,
                    "converged": True,
                    "method": "firth",
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
                }, max_abs_beta)

        return _check_extreme_beta({
            "feature": feature_name,
            "converged": True,
            "method": "standard",
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
        }, max_abs_beta)

    except Exception as e:
        logger.debug("Logistic regression failed for %s: %s", feature_name, e)

        # Attempt Firth as fallback
        if use_firth:
            try:
                firth_res = _fit_firth_logistic(y, X)
                if firth_res is not None and firth_res["converged"]:
                    beta = firth_res["beta"][1]
                    se = firth_res["se"][1]
                    or_val = safe_exp(beta)
                    ci_lower = safe_exp(beta - 1.96 * se)
                    ci_upper = safe_exp(beta + 1.96 * se)
                    z = beta / se if se > 0 else 0.0
                    pvalue = float(2 * sp_stats.norm.sf(abs(z)))
                    return _check_extreme_beta({
                        "feature": feature_name,
                        "converged": True,
                        "method": "firth",
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
                    }, max_abs_beta)
            except Exception:
                pass

        return {
            "feature": feature_name,
            "converged": False,
            "method": "standard",
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
    use_firth: bool = False,
    max_abs_beta: float = 10.0,
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
    use_firth : bool
        Whether to use Firth's penalized regression.

    Returns
    -------
    list of dict
        Results for each feature in the chunk.
    """
    results = []
    for idx in chunk_indices:
        res = _fit_logistic_single(
            dosage_matrix[:, idx], y, X_cov, feature_names[idx], min_carriers,
            use_firth=use_firth, max_abs_beta=max_abs_beta,
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
                use_firth=self.config.use_firth,
                max_abs_beta=self.config.max_abs_beta,
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
                            self.config.min_carriers, self.config.use_firth,
                            self.config.max_abs_beta,
                        )
                    )
        else:
            for chunk in chunks:
                all_results.extend(
                    _process_feature_chunk_risk(
                        chunk, dosage_matrix, y, X_cov, feature_names,
                        self.config.min_carriers, self.config.use_firth,
                        self.config.max_abs_beta,
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
        n_firth = (df["method"] == "firth").sum() if "method" in df.columns else 0
        logger.info(
            "Risk results: dataset=%s, stratum=%s, strategy=%s: "
            "%d features tested, %d significant (FDR < %.2f), %d used Firth",
            dataset_name, stratum, strategy, len(df), n_sig,
            self.config.fdr_threshold, n_firth,
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
            dosage_col, y, X_cov, feature_name, self.config.min_carriers,
            use_firth=self.config.use_firth, max_abs_beta=self.config.max_abs_beta,
        )
