"""
Inverse-variance weighted meta-analysis: fixed and random effects.
"""

import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from hla_analysis.config import AnalysisConfig
from hla_analysis.utils import benjamini_hochberg

logger = logging.getLogger(__name__)


def fixed_effects(betas: np.ndarray, ses: np.ndarray) -> Dict[str, float]:
    """Fixed-effects inverse-variance weighted meta-analysis.

    Parameters
    ----------
    betas : np.ndarray
        Effect estimates from each study.
    ses : np.ndarray
        Standard errors from each study.

    Returns
    -------
    dict
        pooled_beta, pooled_se, ci_lower, ci_upper, pvalue, z_score.
    """
    weights = 1.0 / (ses ** 2)
    total_weight = weights.sum()

    if total_weight < 1e-300:
        return {
            "pooled_beta": np.nan, "pooled_se": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "pvalue": np.nan, "z_score": np.nan,
        }

    pooled_beta = (weights * betas).sum() / total_weight
    pooled_se = 1.0 / np.sqrt(total_weight)
    z = pooled_beta / pooled_se
    pvalue = 2 * sp_stats.norm.sf(abs(z))

    return {
        "pooled_beta": float(pooled_beta),
        "pooled_se": float(pooled_se),
        "ci_lower": float(pooled_beta - 1.96 * pooled_se),
        "ci_upper": float(pooled_beta + 1.96 * pooled_se),
        "pvalue": float(pvalue),
        "z_score": float(z),
    }


def cochran_q(betas: np.ndarray, ses: np.ndarray,
              pooled_beta: float) -> Dict[str, float]:
    """Cochran's Q test for heterogeneity.

    Parameters
    ----------
    betas : np.ndarray
        Study-level effect estimates.
    ses : np.ndarray
        Study-level standard errors.
    pooled_beta : float
        Fixed-effects pooled estimate.

    Returns
    -------
    dict
        q_stat, q_pvalue, i_squared, df.
    """
    k = len(betas)
    weights = 1.0 / (ses ** 2)
    Q = (weights * (betas - pooled_beta) ** 2).sum()
    df = k - 1
    q_pvalue = float(sp_stats.chi2.sf(Q, df)) if df > 0 else np.nan

    # I² statistic
    if df > 0 and Q > 0:
        i_squared = max(0.0, (Q - df) / Q * 100)
    else:
        i_squared = 0.0

    return {
        "q_stat": float(Q),
        "q_pvalue": q_pvalue,
        "i_squared": float(i_squared),
        "df": int(df),
    }


def dersimonian_laird_tau2(betas: np.ndarray, ses: np.ndarray) -> float:
    """DerSimonian-Laird estimator for between-study variance (tau²).

    Parameters
    ----------
    betas : np.ndarray
        Study-level betas.
    ses : np.ndarray
        Study-level standard errors.

    Returns
    -------
    float
        Estimated tau² (non-negative).
    """
    k = len(betas)
    weights = 1.0 / (ses ** 2)
    total_w = weights.sum()

    # Fixed-effects estimate
    pooled = (weights * betas).sum() / total_w

    Q = (weights * (betas - pooled) ** 2).sum()
    df = k - 1

    # C = sum(w_i) - sum(w_i^2) / sum(w_i)
    C = total_w - (weights ** 2).sum() / total_w

    if C < 1e-300:
        return 0.0

    tau2 = max(0.0, (Q - df) / C)
    return float(tau2)


def random_effects(betas: np.ndarray, ses: np.ndarray) -> Dict[str, float]:
    """Random-effects (DerSimonian-Laird) meta-analysis.

    Parameters
    ----------
    betas : np.ndarray
        Study-level betas.
    ses : np.ndarray
        Study-level standard errors.

    Returns
    -------
    dict
        pooled_beta, pooled_se, ci_lower, ci_upper, pvalue, z_score, tau2.
    """
    tau2 = dersimonian_laird_tau2(betas, ses)

    # Random-effects weights
    weights = 1.0 / (ses ** 2 + tau2)
    total_weight = weights.sum()

    if total_weight < 1e-300:
        return {
            "pooled_beta": np.nan, "pooled_se": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "pvalue": np.nan, "z_score": np.nan, "tau2": float(tau2),
        }

    pooled_beta = (weights * betas).sum() / total_weight
    pooled_se = 1.0 / np.sqrt(total_weight)
    z = pooled_beta / pooled_se
    pvalue = 2 * sp_stats.norm.sf(abs(z))

    return {
        "pooled_beta": float(pooled_beta),
        "pooled_se": float(pooled_se),
        "ci_lower": float(pooled_beta - 1.96 * pooled_se),
        "ci_upper": float(pooled_beta + 1.96 * pooled_se),
        "pvalue": float(pvalue),
        "z_score": float(z),
        "tau2": float(tau2),
    }



# Default strategy preference order for best-adjusted meta-analysis
DEFAULT_STRATEGY_PREFERENCE = [
    "all_covariates", "full",
    # drop_* strategies are inserted dynamically
    "reduced", "no_covariates",
]


def create_best_adjusted_results(
    per_dataset_results: pd.DataFrame,
    strategy_preference: Optional[List[str]] = None,
) -> pd.DataFrame:
    """For each dataset × feature × stratum, pick the most-adjusted valid strategy.

    This enables a single meta-analysis across datasets even when different
    datasets succeed with different covariate strategies (e.g. TCGA may only
    have valid results from ``drop_age`` while CIDR works with ``all_covariates``).

    Parameters
    ----------
    per_dataset_results : pd.DataFrame
        Combined per-dataset results with columns including ``dataset``,
        ``feature``, ``stratum``, ``strategy``, ``beta``, ``se``, ``converged``.
    strategy_preference : list of str, optional
        Ordered preference list.  If *None*, uses the default:
        ``all_covariates > full > drop_* (sorted) > reduced > no_covariates``.

    Returns
    -------
    pd.DataFrame
        One row per dataset × feature × stratum, with
        ``strategy='best_adjusted'`` and a ``source_strategy`` column
        indicating which original strategy was selected.
    """
    if per_dataset_results.empty:
        return pd.DataFrame()

    # Build strategy preference order
    if strategy_preference is None:
        # Discover drop_* strategies and sort them
        all_strategies = per_dataset_results["strategy"].unique()
        drop_strategies = sorted(s for s in all_strategies if s.startswith("drop_"))
        strategy_preference = (
            ["all_covariates", "full"]
            + drop_strategies
            + ["reduced", "no_covariates"]
        )

    # Create priority mapping (lower = higher priority)
    priority = {s: i for i, s in enumerate(strategy_preference)}
    # Strategies not in the preference list get the lowest priority
    max_priority = len(strategy_preference)

    # Filter to valid (converged, non-NaN beta/se) results
    valid = per_dataset_results[
        per_dataset_results["converged"].astype(bool) &
        per_dataset_results["beta"].notna() &
        per_dataset_results["se"].notna() &
        (per_dataset_results["se"] > 0)
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    # Assign priority
    valid["_priority"] = valid["strategy"].map(
        lambda s: priority.get(s, max_priority)
    )

    # For each dataset × feature × stratum, pick the row with lowest priority
    group_cols = ["dataset", "feature", "stratum"]
    idx = valid.groupby(group_cols)["_priority"].idxmin()
    best = valid.loc[idx].copy()

    # Tag the result
    best["source_strategy"] = best["strategy"]
    best["strategy"] = "best_adjusted"
    best = best.drop(columns=["_priority"])

    logger.info(
        "Best-adjusted selection: %d rows selected from %d valid results",
        len(best), len(valid),
    )

    return best.reset_index(drop=True)


class MetaAnalyzer:
    """Meta-analysis across datasets for each HLA feature.

    Parameters
    ----------
    config : AnalysisConfig
        Pipeline configuration.
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def run_meta_analysis(
        self,
        results_df: pd.DataFrame,
        effect_col: str = "beta",
        se_col: str = "se",
        analysis_type: str = "risk",
    ) -> pd.DataFrame:
        """Run meta-analysis on per-dataset results.

        Groups results by feature × stratum × strategy, and performs
        both fixed-effects and random-effects meta-analysis.

        Parameters
        ----------
        results_df : pd.DataFrame
            Combined per-dataset results with columns:
            feature, dataset, stratum, strategy, beta, se, pvalue, converged, ...
        effect_col : str
            Column name for effect estimate.
        se_col : str
            Column name for standard error.
        analysis_type : str
            'risk' or 'survival' — used for labelling.

        Returns
        -------
        pd.DataFrame
            Meta-analysis results with one row per feature × stratum × strategy.
        """
        if results_df.empty:
            return pd.DataFrame()

        # Filter to converged results with valid effect + SE
        valid = results_df[
            results_df["converged"] &
            results_df[effect_col].notna() &
            results_df[se_col].notna() &
            (results_df[se_col] > 0)
        ].copy()

        if valid.empty:
            logger.warning("No valid results for meta-analysis (%s)", analysis_type)
            return pd.DataFrame()

        group_cols = ["feature", "stratum", "strategy"]
        meta_results = []

        for group_key, group_df in valid.groupby(group_cols):
            feature, stratum, strategy = group_key
            n_datasets = len(group_df)

            if n_datasets < self.config.meta_min_datasets:
                continue

            betas = group_df[effect_col].values
            ses = group_df[se_col].values
            datasets = group_df["dataset"].tolist()

            # Fixed effects
            fe = fixed_effects(betas, ses)

            # Heterogeneity
            het = cochran_q(betas, ses, fe["pooled_beta"])

            # Random effects
            re = random_effects(betas, ses)

            row = {
                "feature": feature,
                "stratum": stratum,
                "strategy": strategy,
                "analysis_type": analysis_type,
                "n_datasets": n_datasets,
                "datasets": ",".join(datasets),
                # Fixed effects
                "fe_beta": fe["pooled_beta"],
                "fe_se": fe["pooled_se"],
                "fe_ci_lower": fe["ci_lower"],
                "fe_ci_upper": fe["ci_upper"],
                "fe_pvalue": fe["pvalue"],
                "fe_z": fe["z_score"],
                # Random effects
                "re_beta": re["pooled_beta"],
                "re_se": re["pooled_se"],
                "re_ci_lower": re["ci_lower"],
                "re_ci_upper": re["ci_upper"],
                "re_pvalue": re["pvalue"],
                "re_z": re["z_score"],
                "tau2": re.get("tau2", 0.0),
                # Heterogeneity
                "q_stat": het["q_stat"],
                "q_pvalue": het["q_pvalue"],
                "i_squared": het["i_squared"],
            }

            # Add effect size columns appropriate to analysis type
            if analysis_type == "risk":
                row["fe_or"] = float(np.exp(fe["pooled_beta"])) if not np.isnan(fe["pooled_beta"]) else np.nan
                row["fe_or_lower"] = float(np.exp(fe["ci_lower"])) if not np.isnan(fe["ci_lower"]) else np.nan
                row["fe_or_upper"] = float(np.exp(fe["ci_upper"])) if not np.isnan(fe["ci_upper"]) else np.nan
                row["re_or"] = float(np.exp(re["pooled_beta"])) if not np.isnan(re["pooled_beta"]) else np.nan
                row["re_or_lower"] = float(np.exp(re["ci_lower"])) if not np.isnan(re["ci_lower"]) else np.nan
                row["re_or_upper"] = float(np.exp(re["ci_upper"])) if not np.isnan(re["ci_upper"]) else np.nan
            else:
                row["fe_hr"] = float(np.exp(fe["pooled_beta"])) if not np.isnan(fe["pooled_beta"]) else np.nan
                row["fe_hr_lower"] = float(np.exp(fe["ci_lower"])) if not np.isnan(fe["ci_lower"]) else np.nan
                row["fe_hr_upper"] = float(np.exp(fe["ci_upper"])) if not np.isnan(fe["ci_upper"]) else np.nan
                row["re_hr"] = float(np.exp(re["pooled_beta"])) if not np.isnan(re["pooled_beta"]) else np.nan
                row["re_hr_lower"] = float(np.exp(re["ci_lower"])) if not np.isnan(re["ci_lower"]) else np.nan
                row["re_hr_upper"] = float(np.exp(re["ci_upper"])) if not np.isnan(re["ci_upper"]) else np.nan

            # Per-dataset info
            for j, (_, drow) in enumerate(group_df.iterrows()):
                ds = drow["dataset"]
                row[f"{ds}_beta"] = float(drow[effect_col])
                row[f"{ds}_se"] = float(drow[se_col])
                if "pvalue" in drow:
                    row[f"{ds}_pvalue"] = float(drow["pvalue"])

            meta_results.append(row)

        if not meta_results:
            logger.warning("No features had >= %d datasets for meta-analysis",
                           self.config.meta_min_datasets)
            return pd.DataFrame()

        meta_df = pd.DataFrame(meta_results)

        # FDR correction on fixed-effects p-values
        meta_df["fe_fdr"] = benjamini_hochberg(
            meta_df["fe_pvalue"].values, self.config.fdr_threshold
        )
        meta_df["re_fdr"] = benjamini_hochberg(
            meta_df["re_pvalue"].values, self.config.fdr_threshold
        )

        n_fe_sig = (meta_df["fe_fdr"] < self.config.fdr_threshold).sum()
        n_re_sig = (meta_df["re_fdr"] < self.config.fdr_threshold).sum()
        logger.info(
            "Meta-analysis (%s): %d features, %d FE-significant, %d RE-significant (FDR < %.2f)",
            analysis_type, len(meta_df), n_fe_sig, n_re_sig, self.config.fdr_threshold,
        )

        return meta_df


def create_summary_tables(
    risk_meta: pd.DataFrame,
    survival_meta: pd.DataFrame,
    fdr_threshold: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """Create summary tables of significant findings.

    Parameters
    ----------
    risk_meta : pd.DataFrame
        Meta-analysis results for risk analysis.
    survival_meta : pd.DataFrame
        Meta-analysis results for survival analysis.
    fdr_threshold : float
        Significance threshold.

    Returns
    -------
    dict of pd.DataFrame
        Keys: 'risk_significant', 'survival_significant', 'both_significant',
              'by_feature_type', 'by_stratum', 'full_vs_reduced'.
    """
    from hla_analysis.utils import classify_feature

    tables = {}

    # Risk significant
    if not risk_meta.empty and "fe_fdr" in risk_meta.columns:
        risk_sig = risk_meta[risk_meta["fe_fdr"] < fdr_threshold].copy()
        risk_sig["feature_type"] = risk_sig["feature"].apply(classify_feature)
        tables["risk_significant"] = risk_sig.sort_values("fe_pvalue")
    else:
        tables["risk_significant"] = pd.DataFrame()

    # Survival significant
    if not survival_meta.empty and "fe_fdr" in survival_meta.columns:
        surv_sig = survival_meta[survival_meta["fe_fdr"] < fdr_threshold].copy()
        surv_sig["feature_type"] = surv_sig["feature"].apply(classify_feature)
        tables["survival_significant"] = surv_sig.sort_values("fe_pvalue")
    else:
        tables["survival_significant"] = pd.DataFrame()

    # Features significant in BOTH analyses
    if not tables["risk_significant"].empty and not tables["survival_significant"].empty:
        risk_feats = set(
            tables["risk_significant"][["feature", "stratum", "strategy"]].apply(tuple, axis=1)
        )
        surv_feats = set(
            tables["survival_significant"][["feature", "stratum", "strategy"]].apply(tuple, axis=1)
        )
        both = risk_feats & surv_feats
        if both:
            both_df = pd.DataFrame(list(both), columns=["feature", "stratum", "strategy"])
            tables["both_significant"] = both_df
        else:
            tables["both_significant"] = pd.DataFrame()
    else:
        tables["both_significant"] = pd.DataFrame()

    # Summary by feature type and stratum
    for name, meta_df in [("risk", risk_meta), ("survival", survival_meta)]:
        if meta_df.empty or "fe_fdr" not in meta_df.columns:
            continue
        sig = meta_df[meta_df["fe_fdr"] < fdr_threshold].copy()
        if sig.empty:
            continue
        sig["feature_type"] = sig["feature"].apply(classify_feature)

        # By feature type
        by_type = sig.groupby("feature_type").agg(
            n_significant=("feature", "count"),
            min_pvalue=("fe_pvalue", "min"),
            mean_effect=("fe_beta", "mean"),
        ).reset_index()
        tables[f"{name}_by_feature_type"] = by_type

        # By stratum
        by_stratum = sig.groupby("stratum").agg(
            n_significant=("feature", "count"),
            min_pvalue=("fe_pvalue", "min"),
            mean_effect=("fe_beta", "mean"),
        ).reset_index()
        tables[f"{name}_by_stratum"] = by_stratum

    # FULL vs REDUCED comparison
    for name, meta_df in [("risk", risk_meta), ("survival", survival_meta)]:
        if meta_df.empty or "strategy" not in meta_df.columns:
            continue
        full = meta_df[meta_df["strategy"] == "full"][["feature", "stratum", "fe_pvalue", "fe_beta"]].copy()
        reduced = meta_df[meta_df["strategy"] == "reduced"][["feature", "stratum", "fe_pvalue", "fe_beta"]].copy()
        if not full.empty and not reduced.empty:
            comp = full.merge(reduced, on=["feature", "stratum"], suffixes=("_full", "_reduced"))
            tables[f"{name}_full_vs_reduced"] = comp

    return tables
