# HLA Analysis Pipeline

**Production-grade Python package for HLA allele risk and survival association studies with inverse-variance weighted meta-analysis across multiple cohorts.**

## Overview

`hla-analysis` implements a complete HLA allele association pipeline for case-control and survival studies:

- **Risk Analysis**: Logistic regression for HLA allele–disease associations (odds ratios)
- **Survival Analysis**: Cox proportional hazards for HLA allele–outcome associations (hazard ratios)
- **Meta-Analysis**: Fixed-effects and DerSimonian-Laird random-effects meta-analysis across cohorts
- **Visualization**: Manhattan plots, forest plots, heatmaps, and model comparison plots

Designed for HPC environments with support for **multiprocessing parallelism**, **chunked feature processing**, and **memory-aware scheduling**.

## Installation

```bash
# From source
pip install -e .

# With HPC extras (psutil for memory detection)
pip install -e ".[hpc]"

# With development dependencies
pip install -e ".[dev]"

# All optional dependencies
pip install -e ".[all]"
```

### Requirements

- Python ≥ 3.9
- pandas ≥ 1.5, numpy ≥ 1.21, scipy ≥ 1.9
- statsmodels ≥ 0.13 (logistic regression)
- lifelines ≥ 0.27 (Cox PH fallback)
- matplotlib ≥ 3.5, seaborn ≥ 0.12 (visualization)
- tqdm ≥ 4.60 (progress bars)
- Optional: psutil ≥ 5.9 (HPC memory detection)

## Quick Start

```bash
# Full pipeline with 4 datasets
python -m hla_analysis \
  --dosage-files cidr_dosage.csv i370_dosage.csv onco_dosage.csv tcga_dosage.csv \
  --covariate-files cidr_cov.csv i370_cov.csv onco_cov.csv tcga_cov.csv \
  --dataset-names CIDR I370 ONCO TCGA \
  --output-dir results/ \
  --analyses risk survival \
  --workers 64

# Risk only, specific strata
python -m hla_analysis \
  --dosage-files data/*.csv \
  --covariate-files cov/*.csv \
  --analyses risk \
  --strata overall idh_wt \
  --workers 16
```

## Input File Format

### Covariate CSV

| Column | Type | Description |
|--------|------|-------------|
| IID | str | Sample identifier (must match dosage file) |
| dataset | str | Dataset/cohort name |
| age | float | Age at diagnosis |
| sex | str/int | Sex (M/F or 1/0) |
| case | int | Case (1) or control (0) |
| grade | str/int | Tumor grade (HGG/LGG or 1/0) |
| idh | int | IDH mutation status (1=mutant, 0=wild-type) |
| pq | int | 1p/19q co-deletion (1=codel, 0=intact) |
| treated | int | Treatment status (1=treated, 0=untreated) |
| PC1–PC8 | float | Principal components for population stratification |
| survdays | float | Survival time in days (cases only) |
| vstatus | int | Vital status (1=dead, 0=censored) |
| exclude | int | Exclusion flag (1=exclude) |

### HLA Dosage CSV

| Column | Type | Description |
|--------|------|-------------|
| sample_id | str | Sample identifier |
| HLA_A_01 | float | Classical 2-digit allele dosage (0/1/2) |
| HLA_A_01:01 | float | Classical 4-digit allele dosage |
| AA_A_9_V | float | Amino acid variant dosage |

## CLI Parameters

### Input/Output
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dosage-files` | (required) | HLA dosage CSV files |
| `--covariate-files` | (required) | Covariate CSV files |
| `--dataset-names` | auto | Short names for datasets |
| `--output-dir` | `results/` | Output directory |

### Analysis Selection
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--analyses` | `risk survival` | Analyses to run |
| `--strata` | all 5 strata | Strata to analyse |
| `--feature-types` | all types | HLA feature types to include |

### Covariates
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--risk-covariates` | age sex PC1–PC8 | Covariates for logistic regression |
| `--survival-covariates` | age sex grade treated | Covariates for Cox PH |
| `--covariate-strategies` | full reduced | Covariate strategies (ignored when --sensitivity-analysis is set) |
| `--sensitivity-analysis` | off | Enable per-covariate sensitivity analysis |

### Thresholds
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-carriers` | 10 | Min carriers per group (risk) |
| `--min-events` | 5 | Min events among carriers (survival) |
| `--missingness-threshold` | 0.3 | Max missingness for reduced strategy |
| `--fdr-threshold` | 0.05 | FDR significance threshold |
| `--meta-min-datasets` | 2 | Min datasets for meta-analysis |

### HPC / Parallelism
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--workers` | auto (CPU-1) | Parallel workers |
| `--memory-limit` | auto | Memory limit in GB |
| `--chunk-size` | 500 | Features per chunk |

### Cox Solver
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cox-solver` | custom | `custom` (Newton-Raphson) or `lifelines` |
| `--cox-penalizer` | 0.01 | Penalizer for lifelines solver |
| `--cox-max-iter` | 50 | Max iterations (custom solver) |

### Visualization
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--plots` | manhattan forest heatmap | Plot types to generate (also: comparison, sensitivity) |

## Strata Definitions

| Stratum | Description | Case Filter |
|---------|-------------|-------------|
| `overall` | All cases vs all controls | None |
| `idh_wt` | IDH wild-type cases vs controls | idh = 0 |
| `idh_mut` | IDH mutant cases vs controls | idh = 1 |
| `idh_mut_intact` | IDH mut, 1p/19q intact vs controls | idh = 1, pq = 0 |
| `idh_mut_codel` | IDH mut, 1p/19q codel vs controls | idh = 1, pq = 1 |

> **Note**: Controls never have IDH/pq status. All controls are shared across strata.

## Covariate Strategies

- **FULL**: Use all specified covariates; drop samples with any missing values.
- **REDUCED**: Drop covariates with >30% missingness (configurable), then drop samples with remaining missing values. Retains more samples at the cost of fewer adjustment variables.

## Sensitivity Analysis

Enable with `--sensitivity-analysis` (or `--sensitivity`) to see how each covariate with missing data affects your results. When enabled, this **overrides** `--covariate-strategies` and automatically generates:

| Strategy | Covariates Used | Purpose |
|----------|----------------|---------|
| `all_covariates` | All specified covariates | Baseline (same as FULL) |
| `drop_age` | All except age | See impact of dropping age (if age has missingness) |
| `drop_grade` | All except grade | See impact of dropping grade (if grade has missingness) |
| … | … | One strategy per covariate with ANY missing data |
| `no_covariates` | None | Unadjusted model, maximum sample size |

### Example

```bash
# Run risk analysis with sensitivity analysis
python -m hla_analysis \
  --dosage-files cidr.csv i370.csv \
  --covariate-files cidr_cov.csv i370_cov.csv \
  --dataset-names CIDR I370 \
  --analyses risk \
  --sensitivity-analysis \
  --plots manhattan sensitivity \
  --output-dir results/
```

### Key question it answers

> "Does including age (and losing 40% of subjects due to missingness) change the top HLA associations?"

The output `sensitivity_comparison_risk.csv` provides a side-by-side view:

| feature | stratum | dataset | all_covariates_beta | all_covariates_pvalue | drop_age_beta | drop_age_pvalue | no_covariates_beta | no_covariates_pvalue | max_beta_range |
|---------|---------|---------|--------------------|-----------------------|---------------|-----------------|--------------------|----------------------|----------------|
| HLA_A_01 | overall | CIDR | 0.52 | 1.2e-5 | 0.48 | 3.1e-5 | 0.55 | 8.7e-4 | 0.07 |

- **max_beta_range**: Maximum absolute difference in beta across strategies — large values flag features sensitive to covariate choice.
- The sensitivity scatter plots show -log10(p) correlations between baseline and each alternative strategy.

## Output Files

```
results/
├── risk_per_dataset.csv              # Per-dataset logistic regression results
├── survival_per_dataset.csv          # Per-dataset Cox PH results
├── risk_meta_analysis.csv            # Meta-analysis of risk results
├── survival_meta_analysis.csv        # Meta-analysis of survival results
├── sensitivity_comparison_risk.csv   # Side-by-side covariate sensitivity (if --sensitivity-analysis)
├── sensitivity_comparison_survival.csv
├── summary_risk_significant.csv      # Significant risk findings (FDR < threshold)
├── summary_survival_significant.csv
├── summary_both_significant.csv      # Features significant in both analyses
├── pipeline.log                      # Full log file
└── plots/
    ├── manhattan_risk_overall_full.png
    ├── forest_risk_overall_full.png
    ├── heatmap_risk_full.png
    ├── sensitivity_risk.png          # Sensitivity scatter plots (if --sensitivity-analysis)
    └── ...
```

## Meta-Analysis Output Columns

| Column | Description |
|--------|-------------|
| `fe_beta` / `re_beta` | Fixed/random effects pooled log-effect |
| `fe_se` / `re_se` | Pooled standard error |
| `fe_or` / `re_or` | Pooled odds ratio (risk) |
| `fe_hr` / `re_hr` | Pooled hazard ratio (survival) |
| `fe_pvalue` / `re_pvalue` | Pooled p-value |
| `fe_fdr` / `re_fdr` | FDR-adjusted p-value |
| `i_squared` | I² heterogeneity statistic (%) |
| `q_stat` / `q_pvalue` | Cochran's Q test |
| `tau2` | Between-study variance (DerSimonian-Laird) |

## Custom Cox PH Solver

The package includes a fast Newton-Raphson Cox PH solver optimized for single-feature models:

- **Breslow partial likelihood** for tied event times
- **Step-halving** for convergence stability near separation
- **NumPy-only** implementation (faster than lifelines for many single-feature fits)
- Falls back to lifelines if custom solver fails

## Python API

```python
from hla_analysis import AnalysisConfig, DataLoader, RiskAnalyzer, MetaAnalyzer

config = AnalysisConfig(
    dosage_files=["data1.csv", "data2.csv"],
    covariate_files=["cov1.csv", "cov2.csv"],
    dataset_names=["COHORT1", "COHORT2"],
    workers=8,
)

loader = DataLoader(config)
datasets = loader.load_all_datasets()

risk = RiskAnalyzer(config)
# ... run per-dataset analyses ...

meta = MetaAnalyzer(config)
meta_results = meta.run_meta_analysis(combined_results, analysis_type="risk")
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=hla_analysis --cov-report=term-missing

# Run specific test module
python -m pytest tests/test_risk_analysis.py -v

# Skip slow tests
python -m pytest tests/ -v -m "not slow"
```

## Package Structure

```
hla_analysis/
├── hla_analysis/
│   ├── __init__.py          # Version, package metadata
│   ├── __main__.py          # CLI entry point & pipeline orchestration
│   ├── cli.py               # Argument parsing
│   ├── config.py            # Configuration dataclass & defaults
│   ├── data_loader.py       # Load/merge/validate data
│   ├── risk_analysis.py     # Logistic regression (risk)
│   ├── survival_analysis.py # Cox PH (survival)
│   ├── meta_analysis.py     # Fixed/random effects meta-analysis
│   ├── visualization.py     # Manhattan, forest, heatmap, sensitivity plots
│   ├── sensitivity.py       # Sensitivity comparison tables
│   └── utils.py             # Shared utilities, FDR, encoding
├── tests/
│   ├── conftest.py          # Fixtures & synthetic data generators
│   ├── test_data_loader.py
│   ├── test_risk_analysis.py
│   ├── test_survival_analysis.py
│   ├── test_meta_analysis.py
│   ├── test_visualization.py
│   ├── test_cli.py
│   └── test_integration.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── Makefile
```

## License

MIT License
