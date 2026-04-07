"""
Cox proportional hazards survival analysis for HLA allele associations.
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
from hla_analysis.utils import (
    benjamini_hochberg, safe_exp, results_to_dataframe, compute_concordance,
)

logger = logging.getLogger(__name__)


def fast_cox_single(
    time: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> Dict[str, Any]:
    """Fast Cox PH solver using Newton-Raphson with Breslow partial likelihood.

    Handles tied event times using the Breslow approximation.
    Uses step-halving for convergence stability.

    Parameters
    ----------
    time : np.ndarray
        Survival times (n_samples,).
    event : np.ndarray
        Event indicators, 1=event, 0=censored (n_samples,).
    X : np.ndarray
        Design matrix (n_samples, n_covariates). First column is the HLA feature.
    max_iter : int
        Maximum Newton-Raphson iterations.
    tol : float
        Convergence tolerance on log-likelihood change.

    Returns
    -------
    dict
        beta, se, hr, ci_lower, ci_upper, pvalue, concordance, converged, log_likelihood.
    """
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        n, p = X.shape
        beta = np.zeros(p, dtype=np.float64)

        # Sort by time descending for efficient risk-set computation
        sort_idx = np.argsort(-time)
        time_s = time[sort_idx]
        event_s = event[sort_idx]
        X_s = X[sort_idx]

        converged = False
        prev_ll = -np.inf

        for iteration in range(max_iter):
            # Compute exp(X*beta) with overflow protection
            eta = X_s @ beta
            eta = np.clip(eta, -500, 500)
            exp_eta = np.exp(eta)

            # Cumulative sums (over risk sets) — since sorted descending by time,
            # cumsum gives the risk set sums
            cum_exp_eta = np.cumsum(exp_eta)
            cum_X_exp_eta = np.cumsum(X_s * exp_eta[:, np.newaxis], axis=0)
            cum_XX_exp_eta = np.zeros((n, p, p), dtype=np.float64)
            for i in range(n):
                cum_XX_exp_eta[i] = (
                    (cum_XX_exp_eta[i - 1] if i > 0 else np.zeros((p, p)))
                    + np.outer(X_s[i], X_s[i]) * exp_eta[i]
                )

            # Log partial likelihood, gradient, Hessian
            ll = 0.0
            grad = np.zeros(p, dtype=np.float64)
            hess = np.zeros((p, p), dtype=np.float64)

            for i in range(n):
                if event_s[i] != 1:
                    continue
                r = cum_exp_eta[i]
                if r < 1e-300:
                    continue
                rX = cum_X_exp_eta[i]
                rXX = cum_XX_exp_eta[i]

                ll += eta[i] - np.log(r)
                grad += X_s[i] - rX / r
                hess -= rXX / r - np.outer(rX, rX) / (r * r)

            # Check convergence
            if np.isnan(ll) or np.isinf(ll):
                break

            if abs(ll - prev_ll) < tol and iteration > 0:
                converged = True
                break
            prev_ll = ll

            # Newton-Raphson step with step-halving
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                try:
                    step = np.linalg.solve(
                        hess - 1e-6 * np.eye(p), grad
                    )
                except np.linalg.LinAlgError:
                    break

            # Step-halving
            step_size = 1.0
            for _ in range(20):
                beta_new = beta - step_size * step
                eta_new = X_s @ beta_new
                eta_new = np.clip(eta_new, -500, 500)
                exp_eta_new = np.exp(eta_new)
                cum_exp_eta_new = np.cumsum(exp_eta_new)

                ll_new = 0.0
                for i in range(n):
                    if event_s[i] == 1:
                        r = cum_exp_eta_new[i]
                        if r > 1e-300:
                            ll_new += eta_new[i] - np.log(r)

                if ll_new > ll - 1e-10:
                    beta = beta_new
                    break
                step_size *= 0.5
            else:
                beta = beta - step_size * step  # use smallest step

        # Compute standard errors from the observed information matrix
        # Recompute final Hessian
        eta = X_s @ beta
        eta = np.clip(eta, -500, 500)
        exp_eta = np.exp(eta)
        cum_exp_eta = np.cumsum(exp_eta)
        cum_X_exp_eta = np.cumsum(X_s * exp_eta[:, np.newaxis], axis=0)
        cum_XX_exp_eta_final = np.zeros((n, p, p), dtype=np.float64)
        for i in range(n):
            cum_XX_exp_eta_final[i] = (
                (cum_XX_exp_eta_final[i - 1] if i > 0 else np.zeros((p, p)))
                + np.outer(X_s[i], X_s[i]) * exp_eta[i]
            )

        hess_final = np.zeros((p, p), dtype=np.float64)
        for i in range(n):
            if event_s[i] != 1:
                continue
            r = cum_exp_eta[i]
            if r < 1e-300:
                continue
            rX = cum_X_exp_eta[i]
            rXX = cum_XX_exp_eta_final[i]
            hess_final -= rXX / r - np.outer(rX, rX) / (r * r)

        # Variance-covariance matrix
        try:
            var_cov = np.linalg.inv(-hess_final)
            se = np.sqrt(np.maximum(np.diag(var_cov), 0.0))
        except np.linalg.LinAlgError:
            se = np.full(p, np.nan)

        # Results for the HLA feature (first column)
        beta_hla = float(beta[0])
        se_hla = float(se[0])
        hr = safe_exp(beta_hla)
        ci_lower = safe_exp(beta_hla - 1.96 * se_hla) if not np.isnan(se_hla) else np.nan
        ci_upper = safe_exp(beta_hla + 1.96 * se_hla) if not np.isnan(se_hla) else np.nan

        if not np.isnan(se_hla) and se_hla > 0:
            z = beta_hla / se_hla
            pvalue = float(2 * sp_stats.norm.sf(abs(z)))
        else:
            pvalue = np.nan

        # Concordance
        try:
            risk_scores_orig = X @ beta
            conc = compute_concordance(time, event, risk_scores_orig)
        except Exception:
            conc = np.nan

        return {
            "beta": beta_hla,
            "se": se_hla,
            "hr": float(hr),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "pvalue": pvalue,
            "concordance": float(conc),
            "converged": converged,
            "log_likelihood": float(prev_ll),
        }


def _drop_constant_columns(X: np.ndarray, col_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Drop columns with zero variance from the design matrix.

    Parameters
    ----------
    X : np.ndarray
        Design matrix.
    col_names : list of str
        Column names matching X's columns.

    Returns
    -------
    X_filtered : np.ndarray
        Design matrix without constant columns.
    kept_names : list of str
        Column names that were kept.
    """
    if X.shape[1] == 0:
        return X, col_names

    variances = np.var(X, axis=0)
    keep_mask = variances > 1e-10

    if not keep_mask.all():
        dropped = [col_names[i] for i in range(len(col_names)) if not keep_mask[i]]
        for d in dropped:
            logger.warning(
                "Survival analysis: dropping constant covariate '%s' from Cox model", d
            )

    return X[:, keep_mask], [col_names[i] for i in range(len(col_names)) if keep_mask[i]]


def _check_extreme_beta_survival(
    result: Dict[str, Any],
    max_abs_beta: float,
) -> Dict[str, Any]:
    """Post-hoc filter: mark survival results with extreme beta/SE as unstable."""
    beta = result.get("beta", np.nan)
    se = result.get("se", np.nan)
    if (not np.isnan(beta) and abs(beta) > max_abs_beta) or \
       (not np.isnan(se) and se > max_abs_beta):
        logger.debug(
            "Feature %s: extreme survival estimate filtered (|beta|=%.2f, SE=%.2f > %.1f)",
            result.get("feature", "?"), abs(beta) if not np.isnan(beta) else 0,
            se if not np.isnan(se) else 0, max_abs_beta,
        )
        result = dict(result)
        result["beta"] = np.nan
        result["se"] = np.nan
        result["hr"] = np.nan
        result["ci_lower"] = np.nan
        result["ci_upper"] = np.nan
        result["pvalue"] = np.nan
        result["note"] = "filtered_extreme_beta"
    return result


def _fit_survival_single(
    dosage_col: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    X_cov: Optional[np.ndarray],
    feature_name: str,
    min_events: int,
    use_custom: bool = False,
    max_iter: int = 50,
    tol: float = 1e-9,
    penalizer: float = 0.01,
    use_firth: bool = True,
    max_abs_beta: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """Fit a single Cox PH model.

    Parameters
    ----------
    dosage_col : np.ndarray
        Dosage for one HLA feature.
    time : np.ndarray
        Survival times.
    event : np.ndarray
        Event indicators.
    X_cov : np.ndarray or None
        Covariate matrix.
    feature_name : str
        Feature name.
    min_events : int
        Minimum events among carriers.
    use_custom : bool
        If True, use fast_cox_single; else use lifelines.
    max_iter : int
        Max iterations for custom solver.
    tol : float
        Convergence tolerance.
    penalizer : float
        Penalizer for lifelines.
    use_firth : bool
        If True, apply penalization in lifelines CoxPHFitter.

    Returns
    -------
    dict or None
    """
    # Check carrier events
    carrier_mask = dosage_col > 0
    carrier_events = (carrier_mask & (event == 1)).sum()
    if carrier_events < min_events:
        return None

    # Check zero variance
    if np.std(dosage_col) < 1e-10:
        return None

    n_total = len(time)
    n_events = int(event.sum())

    if n_events < 2:
        return None

    # Build design matrix (HLA feature first, then covariates)
    if X_cov is not None:
        X = np.column_stack([dosage_col, X_cov]).astype(np.float64)
    else:
        X = dosage_col.reshape(-1, 1).astype(np.float64)

    if use_custom:
        try:
            result = fast_cox_single(
                time.astype(np.float64),
                event.astype(np.float64),
                X,
                max_iter=max_iter,
                tol=tol,
            )
            result["feature"] = feature_name
            result["n_total"] = n_total
            result["n_events"] = n_events
            result["carrier_events"] = int(carrier_events)
            return _check_extreme_beta_survival(result, max_abs_beta)
        except Exception as e:
            logger.debug("Custom Cox solver failed for %s: %s", feature_name, e)
            # Fall through to lifelines
            use_custom = False

    if not use_custom:
        try:
            from lifelines import CoxPHFitter

            col_names = [feature_name] + [f"cov_{i}" for i in range(X.shape[1] - 1)]
            df = pd.DataFrame(X, columns=col_names)
            df["T"] = time
            df["E"] = event

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Use penalizer when use_firth is True (penalized Cox)
                pen = penalizer if use_firth else 0.0
                cph = CoxPHFitter(penalizer=pen)
                cph.fit(df, duration_col="T", event_col="E")

            summary = cph.summary
            beta = float(summary.loc[feature_name, "coef"])
            se = float(summary.loc[feature_name, "se(coef)"])
            hr = float(summary.loc[feature_name, "exp(coef)"])
            ci_lower = float(summary.loc[feature_name, "exp(coef) lower 95%"])
            ci_upper = float(summary.loc[feature_name, "exp(coef) upper 95%"])
            pvalue = float(summary.loc[feature_name, "p"])
            conc = float(cph.concordance_index_)

            return _check_extreme_beta_survival({
                "feature": feature_name,
                "beta": beta,
                "se": se,
                "hr": hr,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "pvalue": pvalue,
                "concordance": conc,
                "converged": True,
                "n_total": n_total,
                "n_events": n_events,
                "carrier_events": int(carrier_events),
                "log_likelihood": float(cph.log_likelihood_),
            }, max_abs_beta)
        except Exception as e:
            logger.debug("Lifelines Cox PH failed for %s: %s", feature_name, e)
            return {
                "feature": feature_name,
                "beta": np.nan,
                "se": np.nan,
                "hr": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "pvalue": np.nan,
                "concordance": np.nan,
                "converged": False,
                "n_total": n_total,
                "n_events": n_events,
                "carrier_events": int(carrier_events),
                "log_likelihood": np.nan,
            }


def _process_feature_chunk_survival(
    chunk_indices: List[int],
    dosage_matrix: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    X_cov: Optional[np.ndarray],
    feature_names: List[str],
    min_events: int,
    use_custom: bool,
    max_iter: int,
    tol: float,
    penalizer: float,
    use_firth: bool = True,
    max_abs_beta: float = 10.0,
) -> List[Dict[str, Any]]:
    """Process a chunk of features for survival analysis."""
    results = []
    for idx in chunk_indices:
        res = _fit_survival_single(
            dosage_matrix[:, idx], time, event, X_cov,
            feature_names[idx], min_events, use_custom, max_iter, tol,
            penalizer, use_firth, max_abs_beta,
        )
        if res is not None:
            results.append(res)
    return results


class SurvivalAnalyzer:
    """Cox PH survival analysis across features.

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
        time: np.ndarray,
        event: np.ndarray,
        X_cov: Optional[np.ndarray],
        feature_names: List[str],
        dataset_name: str,
        stratum: str,
        strategy: str,
        covariate_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run survival analysis for one dataset-stratum-strategy combination.

        Parameters
        ----------
        dosage_matrix : np.ndarray
            Dosage values (n_samples, n_features).
        time : np.ndarray
            Survival times.
        event : np.ndarray
            Event indicators.
        X_cov : np.ndarray or None
            Covariate matrix.
        feature_names : list of str
            Feature names.
        dataset_name : str
            Dataset name.
        stratum : str
            Stratum name.
        strategy : str
            Covariate strategy.
        covariate_names : list of str, optional
            Covariate names used.

        Returns
        -------
        pd.DataFrame
        """
        n_features = len(feature_names)
        logger.info(
            "Survival analysis: dataset=%s, stratum=%s, strategy=%s, "
            "n_samples=%d, n_features=%d, n_events=%d",
            dataset_name, stratum, strategy, dosage_matrix.shape[0],
            n_features, int(event.sum()),
        )

        if int(event.sum()) < 2:
            logger.warning("Fewer than 2 events; skipping stratum %s", stratum)
            return pd.DataFrame()

        # Drop constant columns from covariate matrix
        if X_cov is not None and X_cov.shape[1] > 0:
            cov_col_names = covariate_names if covariate_names else [f"cov_{i}" for i in range(X_cov.shape[1])]
            X_cov, cov_col_names = _drop_constant_columns(X_cov, cov_col_names)
            covariate_names = cov_col_names
            if X_cov.shape[1] == 0:
                X_cov = None
                logger.warning("All covariates constant; running survival without covariates")

        # Create chunks
        chunk_size = self.config.chunk_size
        chunks = [
            list(range(i, min(i + chunk_size, n_features)))
            for i in range(0, n_features, chunk_size)
        ]

        use_custom = self.config.cox_solver == "custom"

        all_results = []
        workers = min(self.config.workers, len(chunks))

        if workers > 1 and len(chunks) > 1:
            func = partial(
                _process_feature_chunk_survival,
                dosage_matrix=dosage_matrix,
                time=time,
                event=event,
                X_cov=X_cov,
                feature_names=feature_names,
                min_events=self.config.min_events,
                use_custom=use_custom,
                max_iter=self.config.cox_max_iter,
                tol=self.config.cox_tol,
                penalizer=self.config.cox_penalizer,
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
                        _process_feature_chunk_survival(
                            chunk, dosage_matrix, time, event, X_cov,
                            feature_names, self.config.min_events,
                            use_custom, self.config.cox_max_iter,
                            self.config.cox_tol, self.config.cox_penalizer,
                            self.config.use_firth, self.config.max_abs_beta,
                        )
                    )
        else:
            for chunk in chunks:
                all_results.extend(
                    _process_feature_chunk_survival(
                        chunk, dosage_matrix, time, event, X_cov,
                        feature_names, self.config.min_events,
                        use_custom, self.config.cox_max_iter,
                        self.config.cox_tol, self.config.cox_penalizer,
                        self.config.use_firth, self.config.max_abs_beta,
                    )
                )

        if not all_results:
            logger.warning("No features passed filters for survival: dataset=%s, stratum=%s",
                           dataset_name, stratum)
            return pd.DataFrame()

        df = results_to_dataframe(all_results)
        df["dataset"] = dataset_name
        df["stratum"] = stratum
        df["strategy"] = strategy
        if covariate_names:
            df["covariates_used"] = ",".join(covariate_names)

        # FDR correction
        df["fdr"] = benjamini_hochberg(df["pvalue"].values, self.config.fdr_threshold)

        n_sig = (df["fdr"] < self.config.fdr_threshold).sum()
        logger.info(
            "Survival results: dataset=%s, stratum=%s, strategy=%s: "
            "%d features tested, %d significant (FDR < %.2f)",
            dataset_name, stratum, strategy, len(df), n_sig, self.config.fdr_threshold,
        )

        return df

    def run_single_model(
        self,
        dosage_col: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        X_cov: Optional[np.ndarray] = None,
        feature_name: str = "feature",
    ) -> Optional[Dict[str, Any]]:
        """Fit a single Cox PH model (convenience method)."""
        return _fit_survival_single(
            dosage_col, time, event, X_cov, feature_name,
            self.config.min_events,
            use_custom=(self.config.cox_solver == "custom"),
            max_iter=self.config.cox_max_iter,
            tol=self.config.cox_tol,
            penalizer=self.config.cox_penalizer,
            use_firth=self.config.use_firth,
            max_abs_beta=self.config.max_abs_beta,
        )
