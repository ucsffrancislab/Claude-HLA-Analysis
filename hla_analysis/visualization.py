"""
Publication-quality visualizations: Manhattan, forest, heatmap, comparison plots.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from hla_analysis.utils import classify_feature, extract_gene_from_feature

logger = logging.getLogger(__name__)

# HLA gene color palette
HLA_GENE_COLORS = {
    "A": "#E41A1C",
    "B": "#377EB8",
    "C": "#4DAF4A",
    "DRB1": "#984EA3",
    "DQA1": "#FF7F00",
    "DQB1": "#FFFF33",
    "DPA1": "#A65628",
    "DPB1": "#F781BF",
}
DEFAULT_COLOR = "#999999"


class Visualizer:
    """Generate publication-quality plots for HLA analysis results.

    Parameters
    ----------
    output_dir : str
        Directory to save plots.
    fdr_threshold : float
        FDR threshold for significance lines.
    dpi : int
        Figure resolution.
    """

    def __init__(self, output_dir: str = "results", fdr_threshold: float = 0.05,
                 dpi: int = 300):
        self.output_dir = output_dir
        self.fdr_threshold = fdr_threshold
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    def manhattan_plot(
        self,
        results_df: pd.DataFrame,
        analysis_type: str = "risk",
        stratum: str = "overall",
        strategy: str = "full",
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Manhattan-style plot of -log10(p) across HLA features.

        Parameters
        ----------
        results_df : pd.DataFrame
            Meta-analysis or per-dataset results with 'feature' and 'fe_pvalue' columns.
        analysis_type : str
            'risk' or 'survival'.
        stratum : str
            Stratum to plot.
        strategy : str
            Covariate strategy to plot.
        title : str, optional
            Custom title.
        filename : str, optional
            Output filename (without path).

        Returns
        -------
        str
            Path to saved figure.
        """
        if results_df.empty:
            logger.warning("Empty DataFrame for Manhattan plot")
            return ""

        p_col = "fe_pvalue" if "fe_pvalue" in results_df.columns else "pvalue"
        if p_col not in results_df.columns:
            logger.warning("No p-value column found for Manhattan plot")
            return ""

        df = results_df.copy()
        if "stratum" in df.columns:
            df = df[df["stratum"] == stratum]
        if "strategy" in df.columns:
            df = df[df["strategy"] == strategy]

        df = df[df[p_col].notna() & (df[p_col] > 0)].copy()
        if df.empty:
            logger.warning("No data for Manhattan plot: %s, %s, %s",
                           analysis_type, stratum, strategy)
            return ""

        df["neg_log_p"] = -np.log10(df[p_col])
        df["gene"] = df["feature"].apply(extract_gene_from_feature)
        df["color"] = df["gene"].map(HLA_GENE_COLORS).fillna(DEFAULT_COLOR)

        # Sort by gene for grouping
        gene_order = list(HLA_GENE_COLORS.keys()) + sorted(
            set(df["gene"].dropna()) - set(HLA_GENE_COLORS.keys())
        )
        df["gene_order"] = df["gene"].apply(
            lambda g: gene_order.index(g) if g in gene_order else len(gene_order)
        )
        df = df.sort_values(["gene_order", "feature"]).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.scatter(
            range(len(df)), df["neg_log_p"],
            c=df["color"], s=20, alpha=0.7, edgecolors="none",
        )

        # Significance line (Bonferroni)
        bonf = -np.log10(0.05 / len(df)) if len(df) > 0 else 5
        ax.axhline(y=bonf, color="red", linestyle="--", linewidth=0.8, label="Bonferroni")

        # Suggestive line
        ax.axhline(y=-np.log10(1e-3), color="blue", linestyle=":", linewidth=0.6,
                    label="Suggestive (1e-3)")

        # Gene labels on x-axis
        gene_positions = {}
        for g in gene_order:
            mask = df["gene"] == g
            if mask.any():
                positions = df.index[mask]
                gene_positions[g] = (positions[0] + positions[-1]) / 2

        if gene_positions:
            ax.set_xticks(list(gene_positions.values()))
            ax.set_xticklabels(list(gene_positions.keys()), rotation=45, ha="right", fontsize=8)

        # Legend
        handles = [
            mpatches.Patch(color=c, label=g)
            for g, c in HLA_GENE_COLORS.items()
            if g in df["gene"].values
        ]
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)

        ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=12)
        ax.set_xlabel("HLA Region", fontsize=12)

        if title is None:
            effect_label = "OR" if analysis_type == "risk" else "HR"
            title = f"Manhattan Plot — {analysis_type.title()} Analysis ({stratum}, {strategy})"
        ax.set_title(title, fontsize=13, fontweight="bold")

        fig.tight_layout()

        if filename is None:
            filename = f"manhattan_{analysis_type}_{stratum}_{strategy}.png"
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info("Manhattan plot saved: %s", path)
        return path

    def forest_plot(
        self,
        meta_df: pd.DataFrame,
        per_dataset_df: pd.DataFrame,
        analysis_type: str = "risk",
        stratum: str = "overall",
        strategy: str = "full",
        top_n: int = 20,
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Forest plot of top significant alleles with per-dataset and pooled estimates.

        Parameters
        ----------
        meta_df : pd.DataFrame
            Meta-analysis results.
        per_dataset_df : pd.DataFrame
            Per-dataset results.
        analysis_type : str
            'risk' or 'survival'.
        stratum : str
            Stratum to plot.
        strategy : str
            Strategy to plot.
        top_n : int
            Number of top features to show.
        title : str, optional
            Custom title.
        filename : str, optional
            Output filename.

        Returns
        -------
        str
            Path to saved figure.
        """
        # Filter meta results
        mdf = meta_df.copy()
        if "stratum" in mdf.columns:
            mdf = mdf[mdf["stratum"] == stratum]
        if "strategy" in mdf.columns:
            mdf = mdf[mdf["strategy"] == strategy]

        if mdf.empty:
            logger.warning("No meta-analysis data for forest plot")
            return ""

        # Sort by p-value and take top_n
        mdf = mdf.sort_values("fe_pvalue").head(top_n)

        effect_col = "fe_or" if analysis_type == "risk" else "fe_hr"
        lower_col = "fe_or_lower" if analysis_type == "risk" else "fe_hr_lower"
        upper_col = "fe_or_upper" if analysis_type == "risk" else "fe_hr_upper"
        ds_effect_col = "or_val" if analysis_type == "risk" else "hr"
        effect_label = "OR" if analysis_type == "risk" else "HR"

        # Check columns exist
        needed = [effect_col, lower_col, upper_col]
        if not all(c in mdf.columns for c in needed):
            logger.warning("Missing columns for forest plot: %s", needed)
            return ""

        n_features = len(mdf)
        features = mdf["feature"].tolist()

        fig, ax = plt.subplots(figsize=(10, max(4, n_features * 0.5)))

        y_positions = list(range(n_features))

        # Plot per-dataset estimates
        pdf = per_dataset_df.copy()
        if "stratum" in pdf.columns:
            pdf = pdf[pdf["stratum"] == stratum]
        if "strategy" in pdf.columns:
            pdf = pdf[pdf["strategy"] == strategy]

        datasets = sorted(pdf["dataset"].unique()) if "dataset" in pdf.columns else []
        ds_colors = plt.cm.Set2(np.linspace(0, 1, max(len(datasets), 1)))

        for i, feat in enumerate(features):
            y = n_features - 1 - i

            # Per-dataset points
            for j, ds in enumerate(datasets):
                ds_rows = pdf[(pdf["feature"] == feat) & (pdf["dataset"] == ds)]
                if ds_rows.empty:
                    continue
                row = ds_rows.iloc[0]
                if ds_effect_col in row and pd.notna(row[ds_effect_col]):
                    effect = row[ds_effect_col]
                    cl = row.get("ci_lower", np.nan)
                    cu = row.get("ci_upper", np.nan)
                    ax.plot(effect, y + 0.1 * (j - len(datasets) / 2),
                            "o", color=ds_colors[j], markersize=5, alpha=0.7)
                    if pd.notna(cl) and pd.notna(cu):
                        ax.hlines(y + 0.1 * (j - len(datasets) / 2),
                                  cl, cu, colors=ds_colors[j], linewidth=1, alpha=0.5)

            # Meta-analysis diamond
            meta_row = mdf[mdf["feature"] == feat].iloc[0]
            me = meta_row[effect_col]
            ml = meta_row[lower_col]
            mu = meta_row[upper_col]

            if pd.notna(me):
                ax.plot(me, y, "D", color="black", markersize=8, zorder=5)
                if pd.notna(ml) and pd.notna(mu):
                    ax.hlines(y, ml, mu, colors="black", linewidth=2, zorder=5)

        # Reference line at 1
        ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(list(reversed(features)), fontsize=8)
        ax.set_xlabel(f"{effect_label} (95% CI)", fontsize=11)

        if title is None:
            title = f"Forest Plot — Top {n_features} {analysis_type.title()} ({stratum}, {strategy})"
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Legend for datasets
        if datasets:
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ds_colors[j],
                           markersize=6, label=ds)
                for j, ds in enumerate(datasets)
            ]
            handles.append(
                plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
                           markersize=7, label="Meta-analysis")
            )
            ax.legend(handles=handles, loc="lower right", fontsize=7)

        fig.tight_layout()

        if filename is None:
            filename = f"forest_{analysis_type}_{stratum}_{strategy}.png"
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info("Forest plot saved: %s", path)
        return path

    def heatmap_plot(
        self,
        meta_df: pd.DataFrame,
        analysis_type: str = "risk",
        strategy: str = "full",
        top_n: int = 30,
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Heatmap of effect sizes for significant classical alleles across strata.

        Parameters
        ----------
        meta_df : pd.DataFrame
            Meta-analysis results.
        analysis_type : str
            'risk' or 'survival'.
        strategy : str
            Covariate strategy.
        top_n : int
            Max features to show.
        title : str, optional
            Custom title.
        filename : str, optional
            Output filename.

        Returns
        -------
        str
            Path to saved figure.
        """
        effect_col = "fe_beta"

        df = meta_df.copy()
        if "strategy" in df.columns:
            df = df[df["strategy"] == strategy]

        if df.empty:
            logger.warning("No data for heatmap")
            return ""

        # Pivot: features × strata
        pivot = df.pivot_table(
            index="feature", columns="stratum", values=effect_col, aggfunc="first"
        )

        if pivot.empty:
            return ""

        # Select top features by minimum p-value across strata
        min_p = df.groupby("feature")["fe_pvalue"].min()
        top_feats = min_p.nsmallest(top_n).index.tolist()
        pivot = pivot.loc[pivot.index.isin(top_feats)]

        if pivot.empty:
            return ""

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5),
                                         max(4, len(pivot) * 0.35)))

        vmax = np.nanmax(np.abs(pivot.values)) if pivot.values.size > 0 else 1
        sns.heatmap(
            pivot, center=0, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            ax=ax, annot=True, fmt=".2f", linewidths=0.5,
            cbar_kws={"label": "Beta (log-effect)"},
        )

        if title is None:
            title = f"Effect Size Heatmap — {analysis_type.title()} ({strategy})"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("")
        ax.set_xlabel("Stratum")

        fig.tight_layout()

        if filename is None:
            filename = f"heatmap_{analysis_type}_{strategy}.png"
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info("Heatmap saved: %s", path)
        return path

    def comparison_plot(
        self,
        meta_df: pd.DataFrame,
        analysis_type: str = "risk",
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Scatter plot comparing FULL vs REDUCED model p-values.

        Parameters
        ----------
        meta_df : pd.DataFrame
            Meta-analysis results with both strategies.
        analysis_type : str
            'risk' or 'survival'.
        title : str, optional
            Custom title.
        filename : str, optional
            Output filename.

        Returns
        -------
        str
            Path to saved figure.
        """
        df = meta_df.copy()
        if "strategy" not in df.columns:
            return ""

        full = df[df["strategy"] == "full"][["feature", "stratum", "fe_pvalue"]].copy()
        reduced = df[df["strategy"] == "reduced"][["feature", "stratum", "fe_pvalue"]].copy()

        if full.empty or reduced.empty:
            logger.warning("Need both full and reduced strategies for comparison plot")
            return ""

        merged = full.merge(reduced, on=["feature", "stratum"], suffixes=("_full", "_reduced"))
        if merged.empty:
            return ""

        merged["neg_log_p_full"] = -np.log10(merged["fe_pvalue_full"].clip(lower=1e-300))
        merged["neg_log_p_reduced"] = -np.log10(merged["fe_pvalue_reduced"].clip(lower=1e-300))

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(
            merged["neg_log_p_full"], merged["neg_log_p_reduced"],
            s=15, alpha=0.5, edgecolors="none", c="#377EB8",
        )

        # Diagonal
        max_val = max(merged["neg_log_p_full"].max(), merged["neg_log_p_reduced"].max())
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.5)

        # Correlation
        valid = merged.dropna(subset=["neg_log_p_full", "neg_log_p_reduced"])
        if len(valid) > 2:
            from scipy.stats import pearsonr
            r, p = pearsonr(valid["neg_log_p_full"], valid["neg_log_p_reduced"])
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                    fontsize=10, verticalalignment="top")

        ax.set_xlabel(r"$-\log_{10}(p)$ FULL model", fontsize=11)
        ax.set_ylabel(r"$-\log_{10}(p)$ REDUCED model", fontsize=11)

        if title is None:
            title = f"FULL vs REDUCED Covariate Model — {analysis_type.title()}"
        ax.set_title(title, fontsize=12, fontweight="bold")

        fig.tight_layout()

        if filename is None:
            filename = f"comparison_full_vs_reduced_{analysis_type}.png"
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info("Comparison plot saved: %s", path)
        return path

    def sensitivity_plot(
        self,
        comparison_df: pd.DataFrame,
        analysis_type: str = "risk",
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Scatter plots of -log10(p) for all_covariates vs each drop_X strategy.

        Creates a multi-panel figure with one panel per drop_X strategy,
        each showing the relationship between the baseline (all_covariates)
        p-values and the drop_X p-values.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Output of :func:`create_sensitivity_comparison`.  Must have
            columns ``all_covariates_pvalue`` and at least one
            ``drop_*_pvalue`` column.
        analysis_type : str
            ``'risk'`` or ``'survival'``.
        title : str, optional
            Custom overall title.
        filename : str, optional
            Output filename.

        Returns
        -------
        str
            Path to saved figure.
        """
        if comparison_df.empty:
            logger.warning("Empty DataFrame for sensitivity plot")
            return ""

        base_col = "all_covariates_pvalue"
        if base_col not in comparison_df.columns:
            logger.warning("No '%s' column found for sensitivity plot", base_col)
            return ""

        # Identify all drop_* and no_covariates columns
        drop_cols = sorted([
            c for c in comparison_df.columns
            if c.endswith("_pvalue") and c != base_col
        ])

        if not drop_cols:
            logger.warning("No alternative strategy pvalue columns for sensitivity plot")
            return ""

        n_panels = len(drop_cols)
        n_cols = min(n_panels, 3)
        n_rows = (n_panels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4.5 * n_rows),
            squeeze=False,
        )

        for idx, col in enumerate(drop_cols):
            row_i, col_i = divmod(idx, n_cols)
            ax = axes[row_i][col_i]

            df = comparison_df[[base_col, col]].dropna()
            if df.empty:
                ax.set_visible(False)
                continue

            x = -np.log10(df[base_col].clip(lower=1e-300))
            y = -np.log10(df[col].clip(lower=1e-300))

            ax.scatter(x, y, s=12, alpha=0.5, edgecolors="none", c="#377EB8")

            max_val = max(x.max(), y.max(), 1)
            ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.5)

            # Correlation
            if len(df) > 2:
                from scipy.stats import pearsonr
                r, _ = pearsonr(x, y)
                ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                        fontsize=9, verticalalignment="top",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

            strat_label = col.replace("_pvalue", "")
            ax.set_xlabel(r"$-\log_{10}(p)$ all_covariates", fontsize=9)
            ax.set_ylabel(f"$-\\log_{{10}}(p)$ {strat_label}", fontsize=9)
            ax.set_title(strat_label, fontsize=10, fontweight="bold")

        # Hide unused panels
        for idx in range(n_panels, n_rows * n_cols):
            row_i, col_i = divmod(idx, n_cols)
            axes[row_i][col_i].set_visible(False)

        if title is None:
            title = f"Sensitivity Analysis — {analysis_type.title()}"
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()

        if filename is None:
            filename = f"sensitivity_{analysis_type}.png"
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info("Sensitivity plot saved: %s", path)
        return path

