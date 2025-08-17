# Mood Dynamics and Response to Group Music Therapy: Findings from MUSED

This repository contains a fully reproducible, baseline-only analysis of the publicly available [MUSED](https://heidata.uni-heidelberg.de/dataverse/mused) dataset based on [Gaebel et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11929563/) study. The work asks a clinic-facing question: *using only information known at intake*, can baseline data help **(a)** identify participants likely to achieve a large improvement in depressive symptoms with music therapy and **(b)** anticipate post-treatment symptom levels? The pipeline proceeds from data cleaning and EMA summarization, through feature engineering, repeated cross-validated model training, and model interpretability (SHAP, PDPs) and calibration—ending with clear figures and tables suitable for reporting.

---

## Summary of measurements & features in the MUSED dataset

Two source tables are used:

- **Wide (participant-level)** — one row per participant with demographics, diagnoses, and questionnaire scores at **pre** (baseline), **post**, and **follow-up** (when available).
- **Narrow (EMA, momentary-level)** — multiple short mood ratings per day around the **pre** and **post** phases.

### A) Participant-level (“wide”) key variables

| Variable (dataset name) | Type | Timing | Meaning / Notes |
|---|---|---|---|
| `id` | integer | — | Participant identifier |
| `group`, `group_f` | integer / factor | — | Randomized arm: `Control`, `MusicTherapy` |
| `study_cohort` | integer | — | Study wave indicator |
| `age`, `bmi` | numeric | baseline | Demographics |
| `ctq` | numeric | baseline | Childhood Trauma Questionnaire total (as named in dataset) |
| `icd_*` | binary (0/1) | baseline | ICD diagnosis flags (one column per ICD code present in the dataset) |
| `bdi_pre`, `bdi_post`, `bdi_fu` | numeric | pre / post / follow-up | Beck Depression Inventory-II scores |
| `hdrs_pre`, `hdrs_post` | numeric | pre / post | Hamilton Depression Rating Scale scores |
| `hferst_*_pre` | numeric | baseline | HFERST subscales at baseline (dataset labels preserved) |
| `bmmr_*_pre` | numeric | baseline | BMMR subscales at baseline (dataset labels preserved) |
| **Derived:** `bdi_delta_post`, `bdi_delta_fu`, `hdrs_delta_post` | numeric | change | Pre–post/follow-up improvements (positive = improvement when higher scores = worse) |
| **Derived:** `bdi_pct_post`, `bdi_pct_fu`, `hdrs_pct_post` | numeric | change | Percent improvements from baseline |
| **Derived:** `responder_bdi_post`, `responder_bdi_fu`, `responder_hdrs_post` | logical | change | ≥50% reduction indicator (“responder”) |

> When only BMMR subscales were present, a total baseline score `bmmr_tot_pre` was constructed as the row-sum of available subscales (NA-aware).

### B) Momentary-level EMA (“narrow”) key variables and summaries

| Variable (dataset name) | Type | Timing | Meaning / Notes |
|---|---|---|---|
| `id` | integer | — | Participant identifier (links to wide table) |
| `group`, `study_cohort` | integer | — | Arm / wave (copied to EMA rows) |
| `time_pre_post` / `time_pre_post_f` | integer / factor | pre / post | Phase of EMA sampling (`pre`, `post`) |
| `time_day`, `time_moment` | integer | pre / post | Day index and within-day moment index |
| `mom_depress` | numeric | pre / post | Momentary depressed affect rating |

**Baseline EMA summaries** (computed per participant for the **pre** phase and merged into the wide table):

| Summary feature (engineered) | Definition |
|---|---|
| `mom_mean_pre`, `mom_median_pre` | Mean / median of `mom_depress` across all baseline moments |
| `mom_sd_pre` | Within-person variability of `mom_depress` at baseline |
| `slope_mean_pre` | Average within-day linear slope of `mom_depress` at baseline (positive = worsening across the day; negative = improving) |
| `mom_mean_delta_post`, `mom_median_delta_post` | Pre–post differences (where post EMA present) |

---

## Script documentation

### `1_data_processing.R`

**Purpose.** Load the raw MUSED files, harmonize variable names, fix types, and create clinically meaningful derived outcomes and EMA summaries.

**What it does.**
- Creates project folders (`obj/`, `output/`) and reads:
  - `dataverse_files/MUSED_wide_2023_10-02.csv` (participant-level)
  - `dataverse_files/MUSED_momentary_depression_2023_10-02.csv` (EMA)
- Cleans column names (`janitor::clean_names()`), fixes types, and labels randomized group (`group_f`).
- Computes change scores and **responder** flags (≥50% reduction) for BDI/HDRS.
- Aggregates **baseline EMA** to person-level (mean, median, SD, and within-day slope) and joins these to the wide table.
- Produces a long format for repeated outcomes (BDI/HDRS over time).
- Saves processed objects to `obj/` and produces QC plots (trajectories, pre–post change, baseline EMA distributions, and missingness visualization).

**Why this step.** It turns raw tables into a tidy, analysis-ready dataset while summarizing dense EMA into a small set of intuitive baseline features.

---

### `2_feature_engineering.R`

**Purpose.** Define baseline-only predictor sets, create model design matrices with consistent preprocessing, and run baseline statistics (including an ANCOVA).

**What it does.**
- Defines a **baseline-only** feature set: clinical scales at baseline (BDI/HDRS), baseline EMA summaries, demographics (age, BMI), cohort, ICD flags, baseline questionnaire subscales (HFERST/BMMR), and engineered totals (e.g., `bmmr_tot_pre` if subscales only).
- Creates two modeling tables:  
  1) **Classification (MT arm only)** — predict **Responder vs NonResponder** on BDI post.  
  2) **Regression (all participants)** — predict **BDI_post**.
- Builds **tidymodels recipes** to: remove zero-variance predictors, **impute** missing numeric values (k-nearest neighbors), one-hot encode categoricals, **normalize** numerics, and **prune high correlations**.
- Saves baked design matrices to `obj/` for downstream modeling.
- Runs a **collinearity check** (VIF) using reference coding to flag overlapping predictors.
- Fits an **ANCOVA** (linear model) with **BDI_post ~ BDI_pre + EMA_mean_pre + study_cohort + group_f** to quantify the adjusted association between randomized **group** and post-treatment BDI while controlling for baseline severity and cohort.

**Why this step.** It prevents information leakage, standardizes inputs across models, and uses ANCOVA to provide a familiar, adjusted baseline for interpreting treatment-group differences alongside ML results.

---

### `3_modeling_script.R`

**Purpose.** Train and tune three model families for each task under repeated cross-validation, then fit the final models and summarize performance and importances.

**What it does.**
- **Tasks:**  
  • **Classification (MT only):** Elastic Net logistic regression, Random Forest, XGBoost.  
  • **Regression (all participants):** Elastic Net, Random Forest, XGBoost.
- **Resampling:** **5×5 cross-validation** (stratified on outcome for classification).
- **Metrics:**  
  • Classification — ROC AUC, PR AUC, accuracy, sensitivity, specificity.  
  • Regression — RMSE, MAE, R².
- Selects best hyperparameters (ROC AUC for classification, RMSE for regression), **refits** on the full design, and saves final fits to `obj/`.
- Exports coefficient tables (Elastic Net) and variable importance plots (RF/XGB), plus model-performance summaries and a predicted-vs-observed plot for the best regression model.

**Why this step.** Repeated CV provides stable performance estimates that better reflect out-of-sample use. Comparing linear and tree-based models balances interpretability and flexibility.

---

### `4_interpretability_analysis.R`

**Purpose.** Explain how models make predictions, visualize feature effects, and check probability/scale calibration.

**What it does.**
- **SHAP values:**  
  • XGBoost via native `predcontrib`; RF/Elastic Net via `fastshap`.  
  • Produces ranked mean \|SHAP\| tables and top-k bar plots for each model/task.
- **Partial Dependence Plots (PDPs):** Bridges tidymodels predictions to `pdp::partial` and saves PDPs for top SHAP features (classification: P(Responder); regression: predicted BDI_post).
- **Calibration:**  
  • Classification — decile-binned predicted vs observed responder rates (calibration curves).  
  • Regression — predicted vs observed scatter with LOESS and a separate slope/intercept summary.
- **Simplified reporting model (classification):** Baseline logistic regression using only **BDI_pre, HDRS_pre, EMA_mean_pre** with odds ratios and quick performance estimates.

**Why this step.** SHAP highlights which baseline factors matter most; PDPs show the direction and shape of effects in plain terms; calibration checks whether predicted probabilities/values align with reality—key for clinical trust.

---

> **Reproducibility tip.** Run the scripts in order (`1_...` → `4_...`). Intermediate objects are saved under `obj/`, and figures/tables under `output/` (and `output/phase2/` for interpretability).
