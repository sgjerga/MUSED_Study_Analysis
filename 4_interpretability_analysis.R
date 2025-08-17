# 03_phase2_interpretability_mused.R
set.seed(1234)

# ---- packages ----
library(tidyverse)
library(tidymodels)
library(vip)
library(pdp)
library(fastshap)
library(xgboost)
library(probably)
library(glue)
library(broom)

# ---- dirs ----
if (!dir.exists("output")) dir.create("output")
if (!dir.exists("output/phase2")) dir.create("output/phase2")

# ---- load baked designs (from Phase 1 / modeling) ----
cls_obj <- readr::read_rds("obj/cls_mt_design.rds")   # list: data, design, recipe, prep
reg_obj <- readr::read_rds("obj/reg_all_design.rds")

X_cls <- cls_obj$design
X_reg <- reg_obj$design

stopifnot(all(c("id", "responder_bdi_post_f") %in% names(X_cls)))
stopifnot("bdi_post" %in% names(X_reg))

# ---- load final (trained) models & tuning results ----
final_logit_cls <- readRDS("obj/final_logit_cls.rds") # list: res, best, fit
final_rf_cls    <- readRDS("obj/final_rf_cls.rds")
final_xgb_cls   <- readRDS("obj/final_xgb_cls.rds")

final_enet_reg  <- readRDS("obj/final_enet_reg.rds")
final_rf_reg    <- readRDS("obj/final_rf_reg.rds")
final_xgb_reg   <- readRDS("obj/final_xgb_reg.rds")

# Convenience handles (parsnip model_fit objects)
fit_logit_cls <- extract_fit_parsnip(final_logit_cls$fit)
fit_rf_cls    <- extract_fit_parsnip(final_rf_cls$fit)
fit_xgb_cls   <- extract_fit_parsnip(final_xgb_cls$fit)

fit_enet_reg  <- extract_fit_parsnip(final_enet_reg$fit)
fit_rf_reg    <- extract_fit_parsnip(final_rf_reg$fit)
fit_xgb_reg   <- extract_fit_parsnip(final_xgb_reg$fit)

# ---- predictor-only matrices (drop outcome/id) ----
pred_cols_cls <- setdiff(names(X_cls), c("id","responder_bdi_post_f"))
pred_cols_reg <- setdiff(names(X_reg), c("id","bdi_post"))
X_cls_pred <- X_cls[, pred_cols_cls, drop = FALSE]
X_reg_pred <- X_reg[, pred_cols_reg, drop = FALSE]

# Helper: safe gg-save
save_plot <- function(p, path, w=9, h=6, dpi=300) ggsave(path, p, width=w, height=h, dpi=dpi)

# =====================================================
# 1) SHAP VALUES
#    - XGB: native predcontrib
#    - RF/ENet: fastshap with predict wrappers
# =====================================================

## ---- 1A) Classification SHAP ----
# XGB native SHAP for classification (prob of "Responder")
xgb_cls <- fit_xgb_cls$fit
Xmat_cls <- as.matrix(X_cls_pred)
shap_xgb_cls <- as_tibble(predict(xgb_cls, Xmat_cls, predcontrib = TRUE))
bias_col <- if ("BIAS" %in% names(shap_xgb_cls)) "BIAS" else tail(names(shap_xgb_cls), 1)

shap_xgb_cls_long <- shap_xgb_cls |>
  mutate(row = row_number()) |>
  pivot_longer(-row, names_to = "feature", values_to = "shap") |>
  filter(feature != bias_col)

shap_rank_xgb_cls <- shap_xgb_cls_long |>
  group_by(feature) |>
  summarise(mean_abs_shap = mean(abs(shap), na.rm = TRUE), .groups = "drop") |>
  arrange(desc(mean_abs_shap))

readr::write_csv(shap_rank_xgb_cls, "output/phase2/shap_rank_cls_xgb.csv")

p_shap_xgb_cls <- shap_rank_xgb_cls |>
  slice_head(n = 20) |>
  ggplot(aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap)) +
  geom_col() + coord_flip() +
  labs(title = "Classification (MT): XGBoost mean |SHAP| (top 20)",
       x = "Feature", y = "Mean |SHAP|")
save_plot(p_shap_xgb_cls, "output/phase2/cls_xgb_shap_top20.png")

# RF (classification) via fastshap (predict-prob wrapper)
rf_cls_mod <- fit_rf_cls$fit
pred_rf_cls_prob <- function(object, newdata) {
  p <- predict(object, data = newdata, type = "response")$predictions
  if (is.matrix(p)) {
    if ("Responder" %in% colnames(p)) return(as.numeric(p[,"Responder"]))
    return(as.numeric(p[, ncol(p)])) # fallback: last column
  }
  as.numeric(p)
}

set.seed(1)
shap_rf_cls <- fastshap::explain(
  rf_cls_mod,
  X = X_cls_pred,
  pred_wrapper = pred_rf_cls_prob,
  nsim = 200,
  parallel = FALSE
)

shap_rank_rf_cls <- as_tibble(shap_rf_cls) |>
  summarise(across(everything(), ~ mean(abs(.x), na.rm = TRUE))) |>
  pivot_longer(everything(), names_to = "feature", values_to = "mean_abs_shap") |>
  arrange(desc(mean_abs_shap))

readr::write_csv(shap_rank_rf_cls, "output/phase2/shap_rank_cls_rf.csv")

p_shap_rf_cls <- shap_rank_rf_cls |>
  slice_head(n = 20) |>
  ggplot(aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap)) +
  geom_col() + coord_flip() +
  labs(title = "Classification (MT): RF mean |SHAP| (top 20)",
       x = "Feature", y = "Mean |SHAP|")
save_plot(p_shap_rf_cls, "output/phase2/cls_rf_shap_top20.png")

# ENet (classification) via fastshap (glmnet prob wrapper)
logit_cls_mod <- fit_logit_cls$fit
pred_logit_prob <- function(object, newdata) {
  as.numeric(predict(object, newx = as.matrix(newdata), type = "response")[,1])
}

set.seed(2)
shap_logit_cls <- fastshap::explain(
  logit_cls_mod,
  X = X_cls_pred,
  pred_wrapper = pred_logit_prob,
  nsim = 200,
  parallel = FALSE
)

shap_rank_logit_cls <- as_tibble(shap_logit_cls) |>
  summarise(across(everything(), ~ mean(abs(.x), na.rm = TRUE))) |>
  pivot_longer(everything(), names_to = "feature", values_to = "mean_abs_shap") |>
  arrange(desc(mean_abs_shap))

readr::write_csv(shap_rank_logit_cls, "output/phase2/shap_rank_cls_enet.csv")

p_shap_logit_cls <- shap_rank_logit_cls |>
  slice_head(n = 20) |>
  ggplot(aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap)) +
  geom_col() + coord_flip() +
  labs(title = "Classification (MT): ENet mean |SHAP| (top 20)",
       x = "Feature", y = "Mean |SHAP|")
save_plot(p_shap_logit_cls, "output/phase2/cls_enet_shap_top20.png")

## ---- 1B) Regression SHAP ----
xgb_reg <- fit_xgb_reg$fit
Xmat_reg <- as.matrix(X_reg_pred)
shap_xgb_reg <- as_tibble(predict(xgb_reg, Xmat_reg, predcontrib = TRUE))
bias_reg_col <- if ("BIAS" %in% names(shap_xgb_reg)) "BIAS" else tail(names(shap_xgb_reg), 1)

shap_rank_xgb_reg <- shap_xgb_reg |>
  dplyr::select(-all_of(bias_reg_col)) |>
  summarise(across(everything(), ~ mean(abs(.x), na.rm = TRUE))) |>
  pivot_longer(everything(), names_to = "feature", values_to = "mean_abs_shap") |>
  arrange(desc(mean_abs_shap))

readr::write_csv(shap_rank_xgb_reg, "output/phase2/shap_rank_reg_xgb.csv")

p_shap_xgb_reg <- shap_rank_xgb_reg |>
  slice_head(n = 20) |>
  ggplot(aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap)) +
  geom_col() + coord_flip() +
  labs(title = "Regression (All): XGBoost mean |SHAP| (top 20)",
       x = "Feature", y = "Mean |SHAP|")
save_plot(p_shap_xgb_reg, "output/phase2/reg_xgb_shap_top20.png")

# RF regression via fastshap
rf_reg_mod <- fit_rf_reg$fit
pred_rf_reg <- function(object, newdata) {
  as.numeric(predict(object, data = newdata)$predictions)
}

set.seed(3)
shap_rf_reg <- fastshap::explain(
  rf_reg_mod,
  X = X_reg_pred,
  pred_wrapper = pred_rf_reg,
  nsim = 200,
  parallel = FALSE
)

shap_rank_rf_reg <- as_tibble(shap_rf_reg) |>
  summarise(across(everything(), ~ mean(abs(.x), na.rm = TRUE))) |>
  pivot_longer(everything(), names_to = "feature", values_to = "mean_abs_shap") |>
  arrange(desc(mean_abs_shap))

readr::write_csv(shap_rank_rf_reg, "output/phase2/shap_rank_reg_rf.csv")

p_shap_rf_reg <- shap_rank_rf_reg |>
  slice_head(n = 20) |>
  ggplot(aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap)) +
  geom_col() + coord_flip() +
  labs(title = "Regression (All): RF mean |SHAP| (top 20)",
       x = "Feature", y = "Mean |SHAP|")
save_plot(p_shap_rf_reg, "output/phase2/reg_rf_shap_top20.png")

# ENet regression via fastshap
enet_reg_mod <- fit_enet_reg$fit
pred_enet_reg <- function(object, newdata) {
  as.numeric(predict(object, newx = as.matrix(newdata), s = object$lambdaOpt))
}
if (is.null(enet_reg_mod$lambdaOpt)) {
  pred_enet_reg <- function(object, newdata) {
    as.numeric(predict(object, newx = as.matrix(newdata)))
  }
}

set.seed(4)
shap_enet_reg <- fastshap::explain(
  enet_reg_mod,
  X = X_reg_pred,
  pred_wrapper = pred_enet_reg,
  nsim = 200, parallel = FALSE
)

shap_rank_enet_reg <- as_tibble(shap_enet_reg) |>
  summarise(across(everything(), ~ mean(abs(.x), na.rm = TRUE))) |>
  pivot_longer(everything(), names_to = "feature", values_to = "mean_abs_shap") |>
  arrange(desc(mean_abs_shap))

readr::write_csv(shap_rank_enet_reg, "output/phase2/shap_rank_reg_enet.csv")

p_shap_enet_reg <- shap_rank_enet_reg |>
  slice_head(n = 20) |>
  ggplot(aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap)) +
  geom_col() + coord_flip() +
  labs(title = "Regression (All): ENet mean |SHAP| (top 20)",
       x = "Feature", y = "Mean |SHAP|")
save_plot(p_shap_enet_reg, "output/phase2/reg_enet_shap_top20.png")

# ==============================================
# 2) PDPs (patched with a bridge function)
# ==============================================

# One-argument prediction functions (tidymodels interface)
# Classification: P(Responder)
pfun_cls_logit <- function(newdata) {
  as.numeric(predict(fit_logit_cls, new_data = newdata, type = "prob")$.pred_Responder)
}
pfun_cls_rf <- function(newdata) {
  as.numeric(predict(fit_rf_cls, new_data = newdata, type = "prob")$.pred_Responder)
}
pfun_cls_xgb <- function(newdata) {
  as.numeric(predict(fit_xgb_cls, new_data = newdata, type = "prob")$.pred_Responder)
}
# Regression
pfun_reg_enet <- function(newdata) as.numeric(predict(fit_enet_reg, new_data = newdata)$.pred)
pfun_reg_rf   <- function(newdata) as.numeric(predict(fit_rf_reg,   new_data = newdata)$.pred)
pfun_reg_xgb  <- function(newdata) as.numeric(predict(fit_xgb_reg,  new_data = newdata)$.pred)

# Bridge: convert 1-arg pfun to pdp's required (object, newdata)
bridge_pred_fun <- function(pfun_one_arg) {
  force(pfun_one_arg)
  function(object, newdata) pfun_one_arg(newdata)
}

# Generic PDP maker (uses real model object + bridged pred.fun)
make_pdp_fun <- function(model, pfun_one, data, feat,
                         grid.resolution = 20, ylab = "yhat", title_prefix = "") {
  pd <- pdp::partial(
    object = model,
    pred.var = feat,
    train = data,                     # background data (predictor-only)
    pred.fun = bridge_pred_fun(pfun_one),
    grid.resolution = grid.resolution,
    progress = "none"
  )
  # IMPORTANT: pass train=data to autoplot when rug=TRUE
  autoplot(pd, rug = TRUE, train = data) +
    labs(title = glue("{title_prefix} PDP: {feat}"), x = feat, y = ylab)
}

# Top features from SHAP
topk <- function(tbl, k = 6) tbl$feature[seq_len(min(k, nrow(tbl)))]
top_cls_xgb <- topk(shap_rank_xgb_cls, 6)
top_cls_rf  <- topk(shap_rank_rf_cls, 6)
top_cls_en  <- topk(shap_rank_logit_cls, 6)
top_reg_xgb <- topk(shap_rank_xgb_reg, 6)
top_reg_rf  <- topk(shap_rank_rf_reg, 6)
top_reg_en  <- topk(shap_rank_enet_reg, 6)

# Create PDPs (Classification)
for (f in top_cls_xgb) {
  p <- make_pdp_fun(fit_xgb_cls, pfun_cls_xgb, X_cls_pred, f,
                    ylab = "P(Responder)", title_prefix = "XGB (CLS)")
  save_plot(p, glue("output/phase2/cls_xgb_pdp_{f}.png"))
}
for (f in top_cls_rf) {
  p <- make_pdp_fun(fit_rf_cls, pfun_cls_rf, X_cls_pred, f,
                    ylab = "P(Responder)", title_prefix = "RF (CLS)")
  save_plot(p, glue("output/phase2/cls_rf_pdp_{f}.png"))
}
for (f in top_cls_en) {
  p <- make_pdp_fun(fit_logit_cls, pfun_cls_logit, X_cls_pred, f,
                    ylab = "P(Responder)", title_prefix = "ENet (CLS)")
  save_plot(p, glue("output/phase2/cls_enet_pdp_{f}.png"))
}

# Create PDPs (Regression)
for (f in top_reg_xgb) {
  p <- make_pdp_fun(fit_xgb_reg, pfun_reg_xgb, X_reg_pred, f,
                    ylab = "Predicted BDI_post", title_prefix = "XGB (REG)")
  save_plot(p, glue("output/phase2/reg_xgb_pdp_{f}.png"))
}
for (f in top_reg_rf) {
  p <- make_pdp_fun(fit_rf_reg, pfun_reg_rf, X_reg_pred, f,
                    ylab = "Predicted BDI_post", title_prefix = "RF (REG)")
  save_plot(p, glue("output/phase2/reg_rf_pdp_{f}.png"))
}
for (f in top_reg_en) {
  p <- make_pdp_fun(fit_enet_reg, pfun_reg_enet, X_reg_pred, f,
                    ylab = "Predicted BDI_post", title_prefix = "ENet (REG)")
  save_plot(p, glue("output/phase2/reg_enet_pdp_{f}.png"))
}

# ==============================================
# 3) Calibration
# ==============================================

## =========================
## 3A) Classification calibration (binning)
## =========================

# Helper: decile-binned calibration table + plot
calibration_cls <- function(fit, data_X, truth, label, nbins = 10) {
  probs <- predict(fit, new_data = data_X, type = "prob")$.pred_Responder
  df <- tibble(truth = truth, prob = probs) |>
    mutate(bin = ntile(prob, nbins))
  
  cal_tbl <- df |>
    group_by(bin) |>
    summarise(
      mean_pred = mean(prob, na.rm = TRUE),
      obs_rate  = mean(truth == "Responder", na.rm = TRUE),
      n         = n(),
      .groups   = "drop"
    ) |>
    arrange(mean_pred)
  
  p <- ggplot(cal_tbl, aes(x = mean_pred, y = obs_rate)) +
    geom_point(size = 2) +
    geom_line() +
    geom_abline(slope = 1, intercept = 0, linetype = 2) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = TRUE) +
    labs(
      title = glue("Calibration Plot (Classification): {label}"),
      x = "Predicted probability (bin mean)",
      y = "Observed responder rate (bin)"
    ) +
    theme_minimal()
  
  list(table = cal_tbl, plot = p)
}

truth_cls <- X_cls$responder_bdi_post_f

# Build plots/tables
cp_logit <- calibration_cls(fit_logit_cls, X_cls, truth_cls, "Elastic Net")
cp_rf    <- calibration_cls(fit_rf_cls,    X_cls, truth_cls, "Random Forest")
cp_xgb   <- calibration_cls(fit_xgb_cls,   X_cls, truth_cls, "XGBoost")

# Save
readr::write_csv(cp_logit$table, "output/phase2/cls_calibration_enet.csv")
readr::write_csv(cp_rf$table,    "output/phase2/cls_calibration_rf.csv")
readr::write_csv(cp_xgb$table,   "output/phase2/cls_calibration_xgb.csv")

ggsave("output/phase2/cls_calibration_enet.png", cp_logit$plot, width = 7, height = 6, dpi = 300)
ggsave("output/phase2/cls_calibration_rf.png",   cp_rf$plot,    width = 7, height = 6, dpi = 300)
ggsave("output/phase2/cls_calibration_xgb.png",  cp_xgb$plot,   width = 7, height = 6, dpi = 300)

## 3B) Regression calibration (scatter + LOESS)
cal_plot_reg <- function(fit, label) {
  pred <- as.numeric(predict(fit, new_data = X_reg)$.pred)
  df <- tibble(truth = X_reg$bdi_post, pred = pred)
  p <- ggplot(df, aes(x = pred, y = truth)) +
    geom_point(alpha = 0.6) +
    geom_smooth(method = "loess", se = FALSE) +
    geom_abline(slope = 1, intercept = 0, linetype = 2) +
    labs(title = glue("Calibration (Regression): {label}"),
         x = "Predicted BDI_post", y = "Observed BDI_post")
  list(data = df, plot = p)
}

cr_enet <- cal_plot_reg(fit_enet_reg, "Elastic Net")
cr_rf   <- cal_plot_reg(fit_rf_reg,   "Random Forest")
cr_xgb  <- cal_plot_reg(fit_xgb_reg,  "XGBoost")

save_plot(cr_enet$plot, "output/phase2/reg_calibration_enet.png")
save_plot(cr_rf$plot,   "output/phase2/reg_calibration_rf.png")
save_plot(cr_xgb$plot,  "output/phase2/reg_calibration_xgb.png")

readr::write_csv(cr_enet$data, "output/phase2/reg_calibration_enet_points.csv")
readr::write_csv(cr_rf$data,   "output/phase2/reg_calibration_rf_points.csv")
readr::write_csv(cr_xgb$data,  "output/phase2/reg_calibration_xgb_points.csv")

## =========================
## 3B) Regression calibration (slope/intercept + plot)
## =========================

calibration_reg <- function(fit, data_X, truth_vec, label) {
  preds <- as.numeric(predict(fit, new_data = data_X)$.pred)
  df <- tibble(truth = truth_vec, pred = preds)
  
  # “Calibration-in-the-large” (intercept) and slope via linear model
  lm_fit <- lm(truth ~ pred, data = df)
  summ   <- broom::tidy(lm_fit)
  glance <- broom::glance(lm_fit)
  
  # Decile-binned observed vs predicted means (useful summary)
  df$bin <- ntile(df$pred, 10)
  bins <- df |>
    group_by(bin) |>
    summarise(
      mean_pred = mean(pred, na.rm = TRUE),
      mean_obs  = mean(truth, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) |>
    arrange(mean_pred)
  
  p <- ggplot(df, aes(x = pred, y = truth)) +
    geom_point(alpha = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = 2) +
    geom_smooth(method = "lm", se = FALSE) +
    labs(
      title = glue("Calibration (Regression): {label}"),
      subtitle = glue("slope = {round(coef(lm_fit)[2], 3)}, intercept = {round(coef(lm_fit)[1], 3)}"),
      x = "Predicted BDI_post",
      y = "Observed BDI_post"
    ) +
    theme_minimal()
  
  list(lm_summary = summ, lm_glance = glance, bins = bins, plot = p)
}

truth_reg <- X_reg$bdi_post

cr_enet <- calibration_reg(fit_enet_reg, X_reg, truth_reg, "Elastic Net")
cr_rf   <- calibration_reg(fit_rf_reg,   X_reg, truth_reg, "Random Forest")
cr_xgb  <- calibration_reg(fit_xgb_reg,  X_reg, truth_reg, "XGBoost")

# Save tables/plots
readr::write_csv(cr_enet$lm_summary, "output/phase2/reg_calibration_enet_lm.csv")
readr::write_csv(cr_enet$bins,       "output/phase2/reg_calibration_enet_bins.csv")
ggsave("output/phase2/reg_calibration_enet.png", cr_enet$plot, width = 7, height = 6, dpi = 300)

readr::write_csv(cr_rf$lm_summary, "output/phase2/reg_calibration_rf_lm.csv")
readr::write_csv(cr_rf$bins,       "output/phase2/reg_calibration_rf_bins.csv")
ggsave("output/phase2/reg_calibration_rf.png", cr_rf$plot, width = 7, height = 6, dpi = 300)

readr::write_csv(cr_xgb$lm_summary, "output/phase2/reg_calibration_xgb_lm.csv")
readr::write_csv(cr_xgb$bins,       "output/phase2/reg_calibration_xgb_bins.csv")
ggsave("output/phase2/reg_calibration_xgb.png", cr_xgb$plot, width = 7, height = 6, dpi = 300)


# ==============================================
# 4) Simplified models for reporting
# ==============================================

## 4A) Simplified classification (MT only): logistic with core baselines
df_cls_raw <- cls_obj$data

keep_vars_cls <- intersect(
  c("id","responder_bdi_post_f","bdi_pre","hdrs_pre","mom_mean_pre",
    "mom_sd_pre","hferst_rum_pre","hferst_avoid_pre"),
  names(df_cls_raw)
)

df_cls_simpl <- df_cls_raw |>
  dplyr::select(all_of(keep_vars_cls)) |>
  tidyr::drop_na(responder_bdi_post_f, bdi_pre, mom_mean_pre)

simp_logit <- glm(
  responder_bdi_post_f ~ bdi_pre + mom_mean_pre + hdrs_pre,
  data = df_cls_simpl,
  family = binomial()
)

simp_logit_tidy <- broom::tidy(simp_logit, exponentiate = TRUE, conf.int = TRUE)
readr::write_csv(simp_logit_tidy, "output/phase2/simplified_logistic_cls.csv")

# --- Simple performance on full data (optimistic) ---
df_cls_simpl <- df_cls_simpl |>
  mutate(prob_hat = as.numeric(predict(simp_logit, type = "response")),
         pred_cls = if_else(prob_hat >= 0.5, "Responder", "NonResponder") |> factor(levels = levels(responder_bdi_post_f)))

# ROC AUC / PR AUC (note: event is "Responder")
roc_cls_simpl <- yardstick::roc_auc(df_cls_simpl,
                                    truth = responder_bdi_post_f,
                                    prob_hat,
                                    event_level = "second")

pr_cls_simpl  <- yardstick::pr_auc(df_cls_simpl,
                                   truth = responder_bdi_post_f,
                                   prob_hat,
                                   event_level = "second")

acc_cls_simpl <- yardstick::accuracy(df_cls_simpl,
                                     truth = responder_bdi_post_f,
                                     pred_cls)

perf_simpl_tbl <- tibble::tibble(
  metric = c("roc_auc","pr_auc","accuracy"),
  value  = c(roc_cls_simpl$.estimate, pr_cls_simpl$.estimate, acc_cls_simpl$.estimate)
)

readr::write_csv(perf_simpl_tbl, "output/phase2/simplified_logistic_cls_performance.csv")
