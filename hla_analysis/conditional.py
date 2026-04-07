"""
Conditional association analysis for HLA features.

Tests whether the association signal at a target feature is independent
of nearby features (e.g., other alleles at the same gene), by fitting
models that include both the target and each nearby feature as covariates.
"""

import logging
import re
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Gene family groupings for auto-detecting nearby features
_GENE_FAMILIES = {
    "DPA1": ["DPA1", "DPB1"],
    "DPB1": ["DPA1", "DPB1"],
    "DQA1": ["DQA1", "DQB1"],
    "DQB1": ["DQA1", "DQB1"],
    "DRB1": ["DRB1", "DRB3", "DRB4", "DRB5"],
    "DRB3": ["DRB1", "DRB3"],
    "DRB4": ["DRB1", "DRB4"],
    "DRB5": ["DRB1", "DRB5"],
}


def _extract_gene(feature_name: str) -> Optional[str]:
    """Extract the HLA gene name from a feature name.

    Handles both VCF-style (``HLA_A*02:01``) and normalised
    (``HLA_A_02:01``) conventions.

    Returns
    -------
    str or None
        Gene name (e.g. ``"A"``, ``"DRB1"``), or None if not an HLA feature.
    """
    # Try VCF style: HLA_GENE*allele
    m = re.match(r"^HLA_([A-Za-z0-9]+)[*_]", feature_name)
    if m:
        return m.group(1)
    return None


def _get_nearby_features(
    target: str,
    all_features: List[str],
) -> List[str]:
    """Auto-detect nearby features for the target.

    For a target like ``HLA_DPB1_04:01``, returns all other features from
    DPB1 and its family (DPA1).

    Parameters
    ----------
    target : str
        Target feature name.
    all_features : list of str
        All available feature names.

    Returns
    -------
    list of str
        Nearby feature names (excluding the target itself).
    """
    target_gene = _extract_gene(target)
    if target_gene is None:
        return []

    family_genes = _GENE_FAMILIES.get(target_gene, [target_gene])

    nearby = []
    for f in all_features:
        if f == target:
            continue
        gene = _extract_gene(f)
        if gene is not None and gene in family_genes:
            nearby.append(f)

    return nearby


def _fit_logistic(y, X):
    """Fit logistic regression, return (beta_array, pvalue_array) or Nones."""
    import statsmodels.api as sm
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_c = sm.add_constant(X, has_constant="add")
            model = sm.Logit(y, X_c)
            result = model.fit(disp=0, maxiter=100, method="newton")
            # Return params and pvalues for all non-intercept columns
            return result.params[1:], result.pvalues[1:]
    except Exception:
        return None, None


def _fit_cox(time, event, X):
    """Fit Cox PH, return (beta_array, pvalue_array) or Nones."""
    try:
        from lifelines import CoxPHFitter
        col_names = [f"var_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=col_names)
        df["T"] = time
        df["E"] = event
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(df, duration_col="T", event_col="E")
        betas = cph.summary["coef"].values
        pvals = cph.summary["p"].values
        return betas, pvals
    except Exception:
        return None, None


def conditional_analysis(
    dosage_matrix: np.ndarray,
    feature_names: List[str],
    target_feature: str,
    nearby_features: Optional[List[str]] = None,
    y: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    event: Optional[np.ndarray] = None,
    X_cov: Optional[np.ndarray] = None,
    analysis_type: str = "risk",
) -> pd.DataFrame:
    """Run conditional association analysis.

    For each nearby feature, fits two models:
    1. **Unconditional**: nearby feature (+ covariates) only
    2. **Conditional**: nearby feature + target feature (+ covariates)

    If the nearby feature's effect disappears when conditioning on the
    target, the target carries the independent signal.

    Parameters
    ----------
    dosage_matrix : np.ndarray
        Dosage matrix (n_samples, n_features).
    feature_names : list of str
        Feature names corresponding to columns of dosage_matrix.
    target_feature : str
        The feature to condition on.
    nearby_features : list of str, optional
        Features to test against. If None, auto-detected from same gene family.
    y : np.ndarray, optional
        Case/control outcome (required for ``analysis_type="risk"``).
    time : np.ndarray, optional
        Survival times (required for ``analysis_type="survival"``).
    event : np.ndarray, optional
        Event indicators (required for ``analysis_type="survival"``).
    X_cov : np.ndarray, optional
        Covariate matrix.
    analysis_type : str
        ``"risk"`` or ``"survival"``.

    Returns
    -------
    pd.DataFrame
        One row per nearby feature with unconditional and conditional results.
    """
    if target_feature not in feature_names:
        logger.warning("Target feature %s not found in feature list", target_feature)
        return pd.DataFrame()

    target_idx = feature_names.index(target_feature)
    target_dosage = dosage_matrix[:, target_idx].astype(np.float64)

    if nearby_features is None:
        nearby_features = _get_nearby_features(target_feature, feature_names)

    if not nearby_features:
        logger.warning("No nearby features found for %s", target_feature)
        return pd.DataFrame()

    logger.info(
        "Conditional analysis: target=%s, %d nearby features, type=%s",
        target_feature, len(nearby_features), analysis_type,
    )

    results = []
    for nf in nearby_features:
        if nf not in feature_names:
            continue
        nf_idx = feature_names.index(nf)
        nf_dosage = dosage_matrix[:, nf_idx].astype(np.float64)

        # Skip zero-variance features
        if np.std(nf_dosage) < 1e-10 or np.std(target_dosage) < 1e-10:
            continue

        row = {"feature": nf, "target": target_feature}

        if analysis_type == "risk":
            if y is None:
                continue
            # Unconditional: nf only
            X_uncond = nf_dosage.reshape(-1, 1)
            if X_cov is not None:
                X_uncond = np.column_stack([X_uncond, X_cov])
            betas_u, pvals_u = _fit_logistic(y, X_uncond)

            # Conditional: nf + target
            X_cond = np.column_stack([nf_dosage, target_dosage])
            if X_cov is not None:
                X_cond = np.column_stack([X_cond, X_cov])
            betas_c, pvals_c = _fit_logistic(y, X_cond)

        else:  # survival
            if time is None or event is None:
                continue
            X_uncond = nf_dosage.reshape(-1, 1)
            if X_cov is not None:
                X_uncond = np.column_stack([X_uncond, X_cov])
            betas_u, pvals_u = _fit_cox(time, event, X_uncond)

            X_cond = np.column_stack([nf_dosage, target_dosage])
            if X_cov is not None:
                X_cond = np.column_stack([X_cond, X_cov])
            betas_c, pvals_c = _fit_cox(time, event, X_cond)

        # Extract results
        if betas_u is not None:
            row["beta_unconditional"] = float(betas_u[0])
            row["pvalue_unconditional"] = float(pvals_u[0])
        else:
            row["beta_unconditional"] = np.nan
            row["pvalue_unconditional"] = np.nan

        if betas_c is not None:
            row["beta_conditional"] = float(betas_c[0])       # nearby feature
            row["pvalue_conditional"] = float(pvals_c[0])
            row["beta_target_conditional"] = float(betas_c[1])  # target feature
            row["pvalue_target_conditional"] = float(pvals_c[1])
            # Independence: target remains significant when conditioned
            row["independence"] = (
                "independent" if pvals_c[1] < 0.05 else "not_independent"
            )
        else:
            row["beta_conditional"] = np.nan
            row["pvalue_conditional"] = np.nan
            row["beta_target_conditional"] = np.nan
            row["pvalue_target_conditional"] = np.nan
            row["independence"] = "failed"

        results.append(row)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)
