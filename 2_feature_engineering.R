set.seed(1234)

library("tidyverse")
library("tidymodels")
library("janitor")
library("recipes")
library("vip")
library("ggplot2")
library("glmnet")
library("ranger")
library("xgboost")
library("ggrepel")
library("glue")
library("naniar")
library("broom")

## ---- load processed objects from previous script ----
wide_plus   <- readr::read_rds("obj/mused_wide_plus_momentary.rds")
wide_long   <- readr::read_rds("obj/mused_wide_long.rds")
narrow_clean<- readr::read_rds("obj/mused_momentary_clean.rds")

# quick safety: ensure group label present
stopifnot("group_f" %in% names(wide_plus))

## ---- baseline predictor set (no leakage) ----
all_cols <- names(wide_plus)
baseline_cols <- c(
  "bdi_pre","hdrs_pre",
  "mom_mean_pre","mom_median_pre","mom_sd_pre","slope_mean_pre",
  "age","bmi","ctq","study_cohort","group","group_f",
  grep("^icd_", all_cols, value = TRUE),
  grep("^(hferst|bmmr)_[a-z]+_pre$", all_cols, value = TRUE)
) |> unique()
baseline_cols <- intersect(baseline_cols, all_cols)

## ---- targets ----
# Classification: BDI responder (>=50% drop pre->post)
wide_plus <- wide_plus |>
  mutate(
    bdi_pct_post = if_else(!is.na(bdi_pre) & bdi_pre > 0, (bdi_pre - bdi_post)/bdi_pre, NA_real_),
    responder_bdi_post = if_else(!is.na(bdi_pct_post) & bdi_pct_post >= 0.50, 1L, 0L, missing = NA_integer_),
    responder_bdi_post_f = factor(responder_bdi_post, levels = c(0,1), labels = c("NonResponder","Responder"))
  )

# Continuous (diagnostic residuals; keep baseline EMA in the model; use na.exclude)
ancova_refit <- lm(
  bdi_post ~ bdi_pre + mom_mean_pre + study_cohort + group_f,
  data = wide_plus,
  na.action = na.exclude
)
wide_plus$bdipost_resid_adj <- resid(ancova_refit)

## ---- assemble modeling tables ----
# A) Classification: MusicTherapy only
df_cls_mt <- wide_plus |>
  filter(group_f == "MusicTherapy") |>
  dplyr::select(id, group_f, all_of(baseline_cols), responder_bdi_post_f) |>
  filter(!is.na(responder_bdi_post_f))

# B) Regression: all groups (need post & baseline)
df_reg_all <- wide_plus |>
  dplyr::select(id, group_f, all_of(baseline_cols), bdi_post) |>
  filter(!is.na(bdi_post), !is.na(bdi_pre))

## ---- quick diagnostics ----
message("\n=== Row counts ===")
message(glue::glue("Classification (MT only): n = {nrow(df_cls_mt)}"))
message(glue::glue("Regression (all groups): n = {nrow(df_reg_all)}"))

message("\n=== Target balance (classification) ===")
print(df_cls_mt |>
        count(responder_bdi_post_f) |>
        mutate(prop = n/sum(n)))

## ---- missingness screen (handy) ----
miss_pct_cls <- sapply(df_cls_mt, function(z) mean(is.na(z))) |> sort(decreasing = TRUE)
miss_pct_reg <- sapply(df_reg_all, function(z) mean(is.na(z))) |> sort(decreasing = TRUE)
saveRDS(miss_pct_cls, "obj/feat_missingness_cls.rds")
saveRDS(miss_pct_reg, "obj/feat_missingness_reg.rds")
message("\nTop missingness (classification):"); print(head(miss_pct_cls, 10))
message("\nTop missingness (regression):");     print(head(miss_pct_reg, 10))

## ---- tidymodels recipes (baseline EMA included) ----
rec_cls <- recipe(responder_bdi_post_f ~ ., data = df_cls_mt) |>
  update_role(id, new_role = "id") |>
  step_zv(all_predictors()) |>
  step_impute_knn(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors(), -all_outcomes(), one_hot = TRUE) |>
  step_normalize(all_numeric_predictors()) |>
  step_corr(all_numeric_predictors(), threshold = 0.90)

rec_reg <- recipe(bdi_post ~ ., data = df_reg_all) |>
  update_role(id, new_role = "id") |>
  step_zv(all_predictors()) |>
  step_impute_knn(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_normalize(all_numeric_predictors()) |>
  step_corr(all_numeric_predictors(), threshold = 0.90)

## ---- prep & bake (for modeling and for safe diagnostics) ----
prep_cls <- prep(rec_cls)
prep_reg <- prep(rec_reg)

X_cls <- bake(prep_cls, new_data = df_cls_mt) |> as_tibble()
X_reg <- bake(prep_reg, new_data = df_reg_all) |> as_tibble()

readr::write_rds(list(data = df_cls_mt, design = X_cls, recipe = rec_cls, prep = prep_cls),
                 "obj/cls_mt_design.rds")
readr::write_rds(list(data = df_reg_all, design = X_reg, recipe = rec_reg, prep = prep_reg),
                 "obj/reg_all_design.rds")

message("\nSaved:")
message("- obj/cls_mt_design.rds")
message("- obj/reg_all_design.rds")

## ---- ROBUST VIF (reference coding + linear model) ----
# Build a VIF-only recipe using reference coding (no one-hot)
rec_cls_vif <- recipe(responder_bdi_post_f ~ ., data = df_cls_mt) |>
  update_role(id, new_role = "id") |>
  step_zv(all_predictors()) |>
  step_impute_knn(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors(), -all_outcomes(), one_hot = FALSE) |>
  step_normalize(all_numeric_predictors()) |>
  step_corr(all_numeric_predictors(), threshold = 0.90)

prep_vif <- prep(rec_cls_vif)
X_vif <- bake(prep_vif, new_data = df_cls_mt) |> as_tibble() |> dplyr::select(-id)

# Use numeric 0/1 outcome just to build a linear model for VIF
X_vif$y_num <- as.integer(X_vif$responder_bdi_post_f == "Responder")

# Predictors only (drop factor outcome column)
preds_vif <- X_vif |> dplyr::select(-responder_bdi_post_f)

# Extra safety: drop near-zero variance + linear combos
nzv <- caret::nearZeroVar(preds_vif)
if (length(nzv) > 0) preds_vif <- preds_vif[, -nzv, drop = FALSE]
lc <- caret::findLinearCombos(as.matrix(preds_vif))
if (!is.null(lc$remove)) preds_vif <- preds_vif[, -lc$remove, drop = FALSE]

# Fit linear model -> VIF
lm_vif <- lm(y_num ~ ., data = as.data.frame(preds_vif))
vif_vals <- car::vif(lm_vif) |> sort(decreasing = TRUE)
print(head(vif_vals, 15))
saveRDS(vif_vals, "obj/vif_cls.rds")

## ---- prep design matrices & save (post-recipe) ----
prep_cls <- prep(rec_cls)
prep_reg <- prep(rec_reg)

X_cls <- bake(prep_cls, new_data = df_cls_mt) |> as_tibble()
X_reg <- bake(prep_reg, new_data = df_reg_all) |> as_tibble()

# Save design matrices for modeling scripts
readr::write_rds(list(data = df_cls_mt, design = X_cls, recipe = rec_cls, prep = prep_cls),
                 "obj/cls_mt_design.rds")
readr::write_rds(list(data = df_reg_all, design = X_reg, recipe = rec_reg, prep = prep_reg),
                 "obj/reg_all_design.rds")

message("\nSaved:")
message("- obj/cls_mt_design.rds")
message("- obj/reg_all_design.rds")

## ---- quick univariate screens (optional but handy) ----
# AUC for each numeric predictor (classification)
num_preds_cls <- X_cls |> dplyr::select(where(is.numeric))
y_cls <- X_cls$responder_bdi_post_f
auc_tbl <- map_df(setdiff(names(num_preds_cls), "responder_bdi_post_f"), function(v) {
  # simple ROC using yardstick
  d <- tibble(truth = y_cls, pred = num_preds_cls[[v]])
  roc <- yardstick::roc_auc(d, truth, pred, event_level = "second")
  tibble(var = v, auc = roc$.estimate)
}) |> arrange(desc(auc))

readr::write_csv(auc_tbl, "obj/univariate_auc_cls.csv")
print(head(auc_tbl, 15))

## ---- simple permutation importance via ranger (classification) ----
# (This is *not* the final model; just to get a feel for influential features.)
rf_spec <- rand_forest(trees = 1000, mtry = floor(sqrt(ncol(num_preds_cls)-1)), min_n = 5) |>
  set_engine("ranger", importance = "permutation") |>
  set_mode("classification")

wf_rf <- workflow() |> add_recipe(rec_cls) |> add_model(rf_spec)
rf_fit <- fit(wf_rf, df_cls_mt)

vip_tbl <- vip::vi(rf_fit$fit$fit$fit) |> arrange(desc(Importance))
readr::write_csv(vip_tbl, "obj/perm_importance_rf_cls.csv")
print(head(vip_tbl, 20))

# Plot top 20 importance
p_vi <- vip_tbl |> slice_max(Importance, n = 20) |>
  ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() + coord_flip() +
  labs(title = "Permutation importance (Random Forest, MT-only classification)",
       x = "Feature", y = "Importance")
ggsave("output/feat_importance_rf_cls.png", p_vi, width = 10, height = 6, dpi = 300)

## ---- engineered composites (if present) ----
# Examples: compose totals/ratios if subscales exist. Keep *baseline* only.
# BMMR total (already present as bmmr_tot_pre in many datasets; if not, sum)
# If bmmr_*_pre subscales exist but no total, create it now
maybe_bmmr <- grep("^bmmr_.*_pre$", names(wide_plus), value = TRUE)

if (!"bmmr_tot_pre" %in% names(wide_plus)) {
  if (length(maybe_bmmr) > 0) {
    wide_plus <- wide_plus |>
      mutate(bmmr_tot_pre = rowSums(dplyr::select(., all_of(maybe_bmmr)), na.rm = TRUE))
  } else {
    wide_plus <- wide_plus |>
      mutate(bmmr_tot_pre = NA_real_)
  }
}

# Build a compact baseline-only table (no leakage)
baseline_only <- wide_plus |>
  dplyr::select(id, group_f, all_of(baseline_cols), bmmr_tot_pre)

readr::write_rds(baseline_only, "obj/baseline_only_table.rds")
message("\nBaseline-only feature table saved: obj/baseline_only_table.rds")

## ---- print a few “numbers to interpret” ----
message("\n=== Baseline EMA adjustment check (post BDI ANCOVA) ===")
ancova_fit <- lm(bdi_post ~ bdi_pre + mom_mean_pre + study_cohort + group_f, data = df_reg_all)
print(broom::tidy(ancova_fit))
print(broom::glance(ancova_fit))

# Group effect after baseline adjustment:
message("\nAdjusted group effect on BDI_post (reference = Control):")
adj_grp <- broom::tidy(ancova_fit) |> filter(term == "group_fMusicTherapy")
print(adj_grp)

# For classification context (MT only) — baseline drivers among MT:
message("\nCorrelation of baseline predictors with responder (point-biserial, MT only):")
pb_tbl <- df_cls_mt |>
  dplyr::select(responder_bdi_post_f, all_of(baseline_cols)) |>
  mutate(responder = as.integer(responder_bdi_post_f == "Responder")) |>
  dplyr::select(-responder_bdi_post_f) |>
  summarise(across(where(is.numeric), ~ cor(., responder, use = "pairwise.complete.obs"))) |>
  pivot_longer(everything(), names_to = "var", values_to = "point_biserial_corr") |>
  arrange(desc(abs(point_biserial_corr)))
print(head(pb_tbl, 20))
readr::write_csv(pb_tbl, "obj/point_biserial_mt.csv")

message("\nAll done. Use these design files directly in your modeling script.")

