From 9ebb3953a3ca19a659a58667c41309f04ac4f50e Mon Sep 17 00:00:00 2001
From: Operon <operon@local>
Date: Mon, 6 Apr 2026 19:52:15 +0000
Subject: [PATCH] Apply Operon session changes: Firth regression, MAF/R2
 filtering, survival covariates, QQ plots

Key changes:
- config.py: survival covariates rad/chemo (replacing treated), MAF thresholds,
  R2 filtering, Firth regression toggle, QQ plots, split_by_feature_type,
  lifelines Cox solver default
- data_loader.py: compute_maf(), R2 filtering from VCF INFO, MAF-based filtering
  (1% alleles, 0.5% AA), auto-drop all-NaN covariates (TCGA fix)
- risk_analysis.py: Firth penalized logistic regression, expanded output columns,
  feature-type-aware MAF filtering
- survival_analysis.py: lifelines CoxPH integration, improved convergence handling
- cli.py: new CLI flags for Firth, MAF, R2, split-by-feature-type
- __main__.py: end-to-end pipeline with split-by-feature-type support,
  improved orchestration and logging
- visualization.py: QQ plots, improved forest/heatmap/manhattan plots
- __init__.py: minor version bump


