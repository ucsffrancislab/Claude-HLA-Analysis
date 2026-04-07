"""
Publication-quality visualizations: Manhattan, forest, heatmap, QQ, comparison plots.
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
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats as sp_stats

from hla_analysis.utils import classify_feature, extract_gene_from_feature

logger = logging.getLogger(__name__)

# HLA gene color palette — Class I in blue shades, Class II in red/orange
HLA_GENE_COLORS = {
    # Class I — blue shades
    "A": "#1f77b4",
    "B": "#2ca02c",
    "C": "#17becf",
    # Class II — red/orange shades
    "DRB1": "#d62728",
    "DQA1": "#ff7f0e",
    "DQB1": "#e377c2",
    "DPA1": "#8c564b",
    "DPB1": "#bcbd22",
}
CLASS_I_GENES = {"A", "B", "C"}
CLASS_II_GENES = {"DRB1", "DQA1", "DQB1", "DPA1", "DPB1"}
DEFAULT_COLOR = "#999999"


class Visualizer:
    """Generate publication-quality plots for HLA analysis results."""

    def __init__(self, output_dir: str = "results", fdr_threshold: float = 0.05,
                 dpi: int = 300, max_forest_signals: int = 15):
        self.output_dir = output_dir
        self.fdr_threshold = fdr_threshold
        self.dpi = dpi
        self.max_forest_signals = max_forest_signals
        os.makedirs(output_dir, exist_ok=True)

    def manhattan_plot(self, results_df, analysis_type="risk", stratum="overall",
                       strategy="full", title=None, filename=None):
        """Manhattan-style plot with FDR significance line."""
        if results_df.empty:
            return ""
        p_col = "fe_pvalue" if "fe_pvalue" in results_df.columns else "pvalue"
        if p_col not in results_df.columns:
            return ""

        df = results_df.copy()
        if "stratum" in df.columns:
            df = df[df["stratum"] == stratum]
        if "strategy" in df.columns:
            df = df[df["strategy"] == strategy]
        df = df[df[p_col].notna() & (df[p_col] > 0)].copy()
        if df.empty:
            return ""

        df["neg_log_p"] = -np.log10(df[p_col])
        df["gene"] = df["feature"].apply(extract_gene_from_feature)
        df["color"] = df["gene"].map(HLA_GENE_COLORS).fillna(DEFAULT_COLOR)

        gene_order = list(HLA_GENE_COLORS.keys()) + sorted(
            set(df["gene"].dropna()) - set(HLA_GENE_COLORS.keys())
        )
        df["gene_order"] = df["gene"].apply(
            lambda g: gene_order.index(g) if g in gene_order else len(gene_order)
        )
        df = df.sort_values(["gene_order", "feature"]).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.scatter(range(len(df)), df["neg_log_p"], c=df["color"], s=20, alpha=0.7, edgecolors="none")

        # Bonferroni line
        bonf = -np.log10(0.05 / len(df)) if len(df) > 0 else 5
        ax.axhline(y=bonf, color="red", linestyle="--", linewidth=0.8, label="Bonferroni")
        ax.axhline(y=-np.log10(1e-3), color="blue", linestyle=":", linewidth=0.6, label="Suggestive")

        # FDR significance line
        if "fe_fdr" in df.columns:
            fdr_sig = df[df["fe_fdr"] < self.fdr_threshold]
            if not fdr_sig.empty:
                fdr_line = fdr_sig["neg_log_p"].min()
                ax.axhline(y=fdr_line, color="darkgreen", linestyle="--", linewidth=0.8,
                           alpha=0.7, label=f"FDR {self.fdr_threshold}")

        # Gene labels
        gene_positions = {}
        for g in gene_order:
            mask = df["gene"] == g
            if mask.any():
                positions = df.index[mask]
                gene_positions[g] = (positions[0] + positions[-1]) / 2
        if gene_positions:
            ax.set_xticks(list(gene_positions.values()))
            ax.set_xticklabels(list(gene_positions.keys()), rotation=45, ha="right", fontsize=8)

        handles = [mpatches.Patch(color=c, label=g) for g, c in HLA_GENE_COLORS.items() if g in df["gene"].values]
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)

        ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=12)
        ax.set_xlabel("HLA Region", fontsize=12)
        if title is None:
            title = f"Manhattan Plot — {analysis_type.title()} ({stratum}, {strategy})"
        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.tight_layout()

        if filename is None:
            filename = f"manhattan_{analysis_type}_{stratum}_{strategy}.png"
        path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Manhattan plot saved: %s", path)
        return path

    def forest_plot(self, meta_df, per_dataset_df, analysis_type="risk",
                    stratum="overall", strategy="full", top_n=None,
                    title=None, filename=None):
        """Enhanced forest plot with Class I/II coloring, diamonds, and heterogeneity stats."""
        if top_n is None:
            top_n = self.max_forest_signals

        mdf = meta_df.copy()
        if "stratum" in mdf.columns:
            mdf = mdf[mdf["stratum"] == stratum]
        if "strategy" in mdf.columns:
            mdf = mdf[mdf["strategy"] == strategy]
        if mdf.empty:
            return ""

        mdf = mdf.sort_values("fe_pvalue").head(top_n)

        effect_col = "fe_or" if analysis_type == "risk" else "fe_hr"
        lower_col = "fe_or_lower" if analysis_type == "risk" else "fe_hr_lower"
        upper_col = "fe_or_upper" if analysis_type == "risk" else "fe_hr_upper"
        ds_effect_col = "or_val" if analysis_type == "risk" else "hr"
        effect_label = "OR" if analysis_type == "risk" else "HR"

        needed = [effect_col, lower_col, upper_col]
        if not all(c in mdf.columns for c in needed):
            return ""

        n_features = len(mdf)
        features = mdf["feature"].tolist()
        fig_height = max(4, 0.45 * n_features)
        fig, ax = plt.subplots(figsize=(12, fig_height))

        # Light horizontal gridlines
        for i in range(n_features):
            ax.axhline(y=i, color="#f0f0f0", linewidth=0.5, zorder=0)

        pdf = per_dataset_df.copy()
        if "stratum" in pdf.columns:
            pdf = pdf[pdf["stratum"] == stratum]
        if "strategy" in pdf.columns:
            pdf = pdf[pdf["strategy"] == strategy]
        datasets = sorted(pdf["dataset"].unique()) if "dataset" in pdf.columns else []
        ds_colors = plt.cm.Set2(np.linspace(0, 1, max(len(datasets), 1)))

        for i, feat in enumerate(features):
            y = n_features - 1 - i
            gene = extract_gene_from_feature(feat)
            gene_color = HLA_GENE_COLORS.get(gene, DEFAULT_COLOR)

            # Per-dataset squares (sized by 1/SE²)
            for j, ds in enumerate(datasets):
                ds_rows = pdf[(pdf["feature"] == feat) & (pdf["dataset"] == ds)]
                if ds_rows.empty:
                    continue
                row = ds_rows.iloc[0]
                if ds_effect_col in row and pd.notna(row[ds_effect_col]):
                    effect = row[ds_effect_col]
                    cl = row.get("ci_lower", np.nan)
                    cu = row.get("ci_upper", np.nan)
                    se = row.get("se", 1.0)
                    se = max(se, 0.01) if pd.notna(se) else 1.0

                    # Square size proportional to 1/SE²
                    weight = 1.0 / (se ** 2)
                    marker_size = max(3, min(12, weight * 0.5))
                    y_offset = y + 0.08 * (j - len(datasets) / 2)

                    ax.plot(effect, y_offset, "s", color=ds_colors[j],
                            markersize=marker_size, alpha=0.7, zorder=3)

                    if pd.notna(cl) and pd.notna(cu):
                        # Truncate wide CIs
                        log_width = np.log(cu) - np.log(cl) if cl > 0 and cu > 0 else 10
                        if log_width > 5:
                            ax.annotate("", xy=(min(cu, np.exp(np.log(effect) + 2.5)), y_offset),
                                       xytext=(effect, y_offset),
                                       arrowprops=dict(arrowstyle="->", color=ds_colors[j], lw=0.8))
                            ax.annotate("", xy=(max(cl, np.exp(np.log(effect) - 2.5)), y_offset),
                                       xytext=(effect, y_offset),
                                       arrowprops=dict(arrowstyle="->", color=ds_colors[j], lw=0.8))
                        else:
                            ax.hlines(y_offset, cl, cu, colors=ds_colors[j], linewidth=1, alpha=0.5, zorder=2)

            # Meta-analysis diamond
            meta_row = mdf[mdf["feature"] == feat].iloc[0]
            me = meta_row[effect_col]
            ml = meta_row[lower_col]
            mu = meta_row[upper_col]

            if pd.notna(me):
                # Draw diamond for pooled estimate
                diamond_h = 0.2
                diamond_x = [ml, me, mu, me] if pd.notna(ml) and pd.notna(mu) else [me, me, me, me]
                diamond_y = [y, y + diamond_h, y, y - diamond_h]
                ax.fill(diamond_x, diamond_y, color=gene_color, alpha=0.8, zorder=5)
                ax.plot(diamond_x + [diamond_x[0]], diamond_y + [diamond_y[0]],
                       color="black", linewidth=0.5, zorder=6)

            # Heterogeneity annotation
            i_sq = meta_row.get("i_squared", np.nan)
            q_p = meta_row.get("q_pvalue", np.nan)
            fe_p = meta_row.get("fe_pvalue", np.nan)
            stats_text = f"p={fe_p:.1e}" if pd.notna(fe_p) else ""
            if pd.notna(i_sq):
                stats_text += f"  I²={i_sq:.0f}%"
            if stats_text:
                ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 3.0, y,
                       f" {stats_text}", fontsize=8, va="center", color="#555555", zorder=7)

        ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_yticks(list(range(n_features)))
        ax.set_yticklabels(list(reversed(features)), fontsize=10)
        ax.set_xlabel(f"{effect_label} (95% CI)", fontsize=11)

        if title is None:
            title = f"Forest Plot — Top {n_features} {analysis_type.title()} ({stratum}, {strategy})"
        ax.set_title(title, fontsize=12, fontweight="bold")

        if datasets:
            handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=ds_colors[j],
                                  markersize=6, label=ds) for j, ds in enumerate(datasets)]
            handles.append(plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="black",
                                      markersize=7, label="Meta (diamond)"))
            ax.legend(handles=handles, loc="lower right", fontsize=7)

        fig.tight_layout()
        if filename is None:
            filename = f"forest_{analysis_type}_{stratum}_{strategy}.png"
        path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Forest plot saved: %s", path)
        return path

    def qq_plot(self, results_df, analysis_type="risk", stratum="overall",
                strategy="full", title=None, filename=None):
        """QQ plot of observed vs expected -log10(p) with genomic inflation factor."""
        if results_df.empty:
            return ""
        p_col = "fe_pvalue" if "fe_pvalue" in results_df.columns else "pvalue"
        if p_col not in results_df.columns:
            return ""

        df = results_df.copy()
        if "stratum" in df.columns:
            df = df[df["stratum"] == stratum]
        if "strategy" in df.columns:
            df = df[df["strategy"] == strategy]
        df = df[df[p_col].notna() & (df[p_col] > 0)].copy()
        if df.empty:
            return ""

        observed_p = np.sort(df[p_col].values)
        n = len(observed_p)
        expected_p = (np.arange(1, n + 1) - 0.5) / n

        obs_log = -np.log10(observed_p)
        exp_log = -np.log10(expected_p)

        # Genomic inflation factor
        chi2_obs = sp_stats.chi2.ppf(1 - observed_p, 1)
        chi2_obs = chi2_obs[np.isfinite(chi2_obs)]
        lambda_gc = float(np.median(chi2_obs) / 0.4549364) if len(chi2_obs) > 0 else 1.0

        # Color by gene
        df_sorted = df.sort_values(p_col).reset_index(drop=True)
        genes = df_sorted["feature"].apply(extract_gene_from_feature)
        colors = genes.map(HLA_GENE_COLORS).fillna(DEFAULT_COLOR).values

        fig, ax = plt.subplots(figsize=(7, 7))

        # 95% confidence band
        ci_upper = sp_stats.beta.ppf(0.975, np.arange(1, n + 1), np.arange(n, 0, -1))
        ci_lower = sp_stats.beta.ppf(0.025, np.arange(1, n + 1), np.arange(n, 0, -1))
        ax.fill_between(exp_log, -np.log10(ci_upper), -np.log10(ci_lower),
                        alpha=0.15, color="gray", label="95% CI")

        # Diagonal
        max_val = max(exp_log.max(), obs_log.max()) * 1.05
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.5, label="y = x")

        # Points colored by gene
        ax.scatter(exp_log, obs_log, c=colors, s=15, alpha=0.7, edgecolors="none", zorder=3)

        # Lambda annotation
        ax.text(0.05, 0.95, f"$\lambda_{{GC}}$ = {lambda_gc:.3f}\nn = {n}",
                transform=ax.transAxes, fontsize=11, verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.3"))

        ax.set_xlabel(r"Expected $-\log_{10}(p)$", fontsize=12)
        ax.set_ylabel(r"Observed $-\log_{10}(p)$", fontsize=12)
        if title is None:
            title = f"QQ Plot — {analysis_type.title()} ({stratum}, {strategy})"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)

        # Gene legend
        handles = [mpatches.Patch(color=c, label=g) for g, c in HLA_GENE_COLORS.items()
                   if g in genes.values]
        if handles:
            ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)

        fig.tight_layout()
        if filename is None:
            filename = f"qq_{analysis_type}_{stratum}_{strategy}.png"
        path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("QQ plot saved: %s", path)
        return path

    def heatmap_plot(self, meta_df, analysis_type="risk", strategy="full",
                     top_n=30, title=None, filename=None):
        """Heatmap of effect sizes across strata."""
        effect_col = "fe_beta"
        df = meta_df.copy()
        if "strategy" in df.columns:
            df = df[df["strategy"] == strategy]
        if df.empty:
            return ""

        pivot = df.pivot_table(index="feature", columns="stratum", values=effect_col, aggfunc="first")
        if pivot.empty:
            return ""

        min_p = df.groupby("feature")["fe_pvalue"].min()
        top_feats = min_p.nsmallest(top_n).index.tolist()
        pivot = pivot.loc[pivot.index.isin(top_feats)]
        if pivot.empty:
            return ""

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.35)))
        vmax = np.nanmax(np.abs(pivot.values)) if pivot.values.size > 0 else 1
        sns.heatmap(pivot, center=0, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    ax=ax, annot=True, fmt=".2f", linewidths=0.5,
                    cbar_kws={"label": "Beta (log-effect)"})
        if title is None:
            title = f"Effect Size Heatmap — {analysis_type.title()} ({strategy})"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("")
        ax.set_xlabel("Stratum")
        fig.tight_layout()

        if filename is None:
            filename = f"heatmap_{analysis_type}_{strategy}.png"
        path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Heatmap saved: %s", path)
        return path

    def comparison_plot(self, meta_df, analysis_type="risk", title=None, filename=None):
        """Scatter plot comparing FULL vs REDUCED model p-values."""
        df = meta_df.copy()
        if "strategy" not in df.columns:
            return ""
        full = df[df["strategy"] == "full"][["feature", "stratum", "fe_pvalue"]].copy()
        reduced = df[df["strategy"] == "reduced"][["feature", "stratum", "fe_pvalue"]].copy()
        if full.empty or reduced.empty:
            return ""

        merged = full.merge(reduced, on=["feature", "stratum"], suffixes=("_full", "_reduced"))
        if merged.empty:
            return ""

        merged["neg_log_p_full"] = -np.log10(merged["fe_pvalue_full"].clip(lower=1e-300))
        merged["neg_log_p_reduced"] = -np.log10(merged["fe_pvalue_reduced"].clip(lower=1e-300))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(merged["neg_log_p_full"], merged["neg_log_p_reduced"],
                   s=15, alpha=0.5, edgecolors="none", c="#377EB8")

        max_val = max(merged["neg_log_p_full"].max(), merged["neg_log_p_reduced"].max())
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.5)

        valid = merged.dropna(subset=["neg_log_p_full", "neg_log_p_reduced"])
        if len(valid) > 2:
            from scipy.stats import pearsonr
            r, p = pearsonr(valid["neg_log_p_full"], valid["neg_log_p_reduced"])
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment="top")

        ax.set_xlabel(r"$-\log_{10}(p)$ FULL model", fontsize=11)
        ax.set_ylabel(r"$-\log_{10}(p)$ REDUCED model", fontsize=11)
        if title is None:
            title = f"FULL vs REDUCED — {analysis_type.title()}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.tight_layout()

        if filename is None:
            filename = f"comparison_full_vs_reduced_{analysis_type}.png"
        path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Comparison plot saved: %s", path)
        return path

    def sensitivity_plot(self, comparison_df, analysis_type="risk", title=None, filename=None):
        """Sensitivity analysis scatter plots."""
        if comparison_df.empty:
            return ""
        base_col = "all_covariates_pvalue"
        if base_col not in comparison_df.columns:
            return ""

        drop_cols = sorted([c for c in comparison_df.columns if c.endswith("_pvalue") and c != base_col])
        if not drop_cols:
            return ""

        n_panels = len(drop_cols)
        n_cols = min(n_panels, 3)
        n_rows = (n_panels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)

        for idx, col in enumerate(drop_cols):
            row_i, col_i = divmod(idx, n_cols)
            ax = axes[row_i][col_i]
            df = comparison_df[[base_col, col]].dropna()
            if df.empty:
                ax.set_visible(False)
                continue
            x = -np.log10(df[base_col].clip(lower=1e-300))
            y_vals = -np.log10(df[col].clip(lower=1e-300))
            ax.scatter(x, y_vals, s=12, alpha=0.5, edgecolors="none", c="#377EB8")
            max_val = max(x.max(), y_vals.max(), 1)
            ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.5)
            if len(df) > 2:
                from scipy.stats import pearsonr
                r, _ = pearsonr(x, y_vals)
                ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes, fontsize=9,
                        verticalalignment="top", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
            strat_label = col.replace("_pvalue", "")
            ax.set_xlabel(r"$-\log_{10}(p)$ all_covariates", fontsize=9)
            ax.set_ylabel(f"$-\\log_{{10}}(p)$ {strat_label}", fontsize=9)
            ax.set_title(strat_label, fontsize=10, fontweight="bold")

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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Sensitivity plot saved: %s", path)
        return path
