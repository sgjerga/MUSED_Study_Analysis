set.seed(1234)

library("tidyverse")
library("tidymodels")
library("vip")
library("glmnet")
library("ranger")
library("xgboost")
library("ggplot2")
library("glue")
library("broom")

# ---- load baked designs from 02_feature_engineering_mused.R ----
cls_obj <- readr::read_rds("obj/cls_mt_design.rds")  # list: data, design, recipe, prep
reg_obj <- readr::read_rds("obj/reg_all_design.rds")

df_cls   <- cls_obj$data
X_cls    <- cls_obj$design
df_reg   <- reg_obj$data
X_reg    <- reg_obj$design

# Keep only columns we actually need (drop any extraneous roles)
# Classification: outcome is responder_bdi_post_f
stopifnot("responder_bdi_post_f" %in% names(X_cls))
# Regression: outcome is bdi_post
stopifnot("bdi_post" %in% names(X_reg))

# ---- resampling ----
# Classification: stratified repeated vfold CV
set.seed(1)
folds_cls <- vfold_cv(X_cls, v = 5, repeats = 5, strata = responder_bdi_post_f)

# Regression: repeated vfold (no stratification)
set.seed(2)
folds_reg <- vfold_cv(X_reg, v = 5, repeats = 5)

# ---- metrics ----
metric_set_cls <- metric_set(roc_auc, pr_auc, accuracy, sens, spec)
metric_set_reg <- metric_set(rmse, mae, rsq)

# =========================
# = CLASSIFICATION (MT)   =
# =========================

# Models
logit_spec <- logistic_reg(
  penalty = tune(), mixture = tune()  # elastic net
) %>% set_engine("glmnet")

rf_spec <- rand_forest(
  mtry  = tune(), trees = 1000, min_n = tune()
) %>% set_engine("ranger", importance = "permutation") %>% set_mode("classification")

xgb_spec <- boost_tree(
  trees = tune(), tree_depth = tune(), learn_rate = tune(),
  mtry  = tune(), min_n = tune(), loss_reduction = tune()
) %>% set_engine("xgboost") %>% set_mode("classification")

# Preprocessing: use the baked design directly (already cleaned/normalized)
# We still wrap a trivial recipe to let tidymodels tune consistently
rec_cls_final <- recipe(responder_bdi_post_f ~ ., data = X_cls) %>%
  update_role(id, new_role = "id") %>%
  step_zv(all_predictors())

# Grids
logit_grid <- grid_regular(
  penalty(range = c(-5, 1)), # 1e-5 to 10
  mixture(range = c(0, 1)),
  levels = 7
)

rf_grid <- grid_random(
  mtry(range = c(5L, max(5L, floor(ncol(X_cls)/2)))),
  min_n(range = c(2L, 10L)),
  size = 20
)

xgb_grid <- grid_random(
  trees(range = c(200L, 1500L)),
  tree_depth(range = c(2L, 6L)),
  learn_rate(range = c(0.01, 0.3)),
  mtry(range = c(5L, max(5L, floor(ncol(X_cls)/2)))),
  min_n(range = c(2L, 20L)),
  loss_reduction(),
  size = 30
)

# Workflows
wf_logit_cls <- workflow() %>% add_recipe(rec_cls_final) %>% add_model(logit_spec)
wf_rf_cls    <- workflow() %>% add_recipe(rec_cls_final) %>% add_model(rf_spec)
wf_xgb_cls   <- workflow() %>% add_recipe(rec_cls_final) %>% add_model(xgb_spec)

# Tuning (optimize ROC AUC)
ctrl <- control_grid(save_pred = TRUE, save_workflow = TRUE)

res_logit_cls <- tune_grid(
  wf_logit_cls, resamples = folds_cls, grid = logit_grid,
  metrics = metric_set_cls, control = ctrl
)
res_rf_cls <- tune_grid(
  wf_rf_cls, resamples = folds_cls, grid = rf_grid,
  metrics = metric_set_cls, control = ctrl
)
res_xgb_cls <- tune_grid(
  wf_xgb_cls, resamples = folds_cls, grid = xgb_grid,
  metrics = metric_set_cls, control = ctrl
)

# Select best by ROC AUC
best_logit_cls <- select_best(res_logit_cls, metric = "roc_auc")
best_rf_cls    <- select_best(res_rf_cls,    metric = "roc_auc")
best_xgb_cls   <- select_best(res_xgb_cls,   metric = "roc_auc")

# Collect metrics
summ_logit_cls <- collect_metrics(res_logit_cls) %>% filter(.metric %in% c("roc_auc","pr_auc","accuracy"))
summ_rf_cls    <- collect_metrics(res_rf_cls)    %>% filter(.metric %in% c("roc_auc","pr_auc","accuracy"))
summ_xgb_cls   <- collect_metrics(res_xgb_cls)   %>% filter(.metric %in% c("roc_auc","pr_auc","accuracy"))

# Save tables
readr::write_csv(summ_logit_cls, "output/cv_cls_logit_metrics.csv")
readr::write_csv(summ_rf_cls,    "output/cv_cls_rf_metrics.csv")
readr::write_csv(summ_xgb_cls,   "output/cv_cls_xgb_metrics.csv")

# Finalize and fit on full X_cls
final_logit_cls <- finalize_workflow(wf_logit_cls, best_logit_cls) %>% fit(X_cls)
final_rf_cls    <- finalize_workflow(wf_rf_cls, best_rf_cls)       %>% fit(X_cls)
final_xgb_cls   <- finalize_workflow(wf_xgb_cls, best_xgb_cls)     %>% fit(X_cls)

saveRDS(list(res = res_logit_cls, best = best_logit_cls, fit = final_logit_cls),
        "obj/final_logit_cls.rds")
saveRDS(list(res = res_rf_cls, best = best_rf_cls, fit = final_rf_cls),
        "obj/final_rf_cls.rds")
saveRDS(list(res = res_xgb_cls, best = best_xgb_cls, fit = final_xgb_cls),
        "obj/final_xgb_cls.rds")

# Feature importance / coefficients
# Elastic net: coefficients
logit_coefs <- tidy(extract_fit_parsnip(final_logit_cls)) %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(estimate)))
readr::write_csv(logit_coefs, "output/cls_logit_coefficients.csv")

# RF: permutation importance
rf_vip <- vip::vi(extract_fit_parsnip(final_rf_cls)$fit) %>% arrange(desc(Importance))
readr::write_csv(rf_vip, "output/cls_rf_importance.csv")
p_rf <- rf_vip %>% slice_max(Importance, n = 20) %>%
  ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() + coord_flip() +
  labs(title = "Classification (MT): Random Forest Permutation Importance",
       x = "Feature", y = "Importance")
ggsave("output/cls_rf_importance_top20.png", p_rf, width = 9, height = 6, dpi = 300)

# XGB: gain importance
xgb_fit_obj <- extract_fit_parsnip(final_xgb_cls)$fit
xgb_imp <- xgboost::xgb.importance(model = xgb_fit_obj)
readr::write_csv(as_tibble(xgb_imp), "output/cls_xgb_importance.csv")
p_xgb <- as_tibble(xgb_imp) %>% slice_max(Gain, n = 20) %>%
  ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col() + coord_flip() +
  labs(title = "Classification (MT): XGBoost Gain Importance",
       x = "Feature", y = "Gain")
ggsave("output/cls_xgb_importance_top20.png", p_xgb, width = 9, height = 6, dpi = 300)

# Sanity: best cross-validated metrics (mean over resamples)
cv_best_cls <- bind_rows(
  collect_metrics(res_logit_cls) %>% mutate(model = "logit"),
  collect_metrics(res_rf_cls)    %>% mutate(model = "rf"),
  collect_metrics(res_xgb_cls)   %>% mutate(model = "xgb")
) %>%
  group_by(model, .metric) %>%
  summarise(mean = mean(mean), std_err = mean(std_err), .groups = "drop") %>%
  arrange(desc(ifelse(.metric=="roc_auc", mean, -Inf)))
readr::write_csv(cv_best_cls, "output/cls_cv_summary.csv")

# =======================
# = REGRESSION (All)    =
# =======================

# Models
enet_reg <- linear_reg(
  penalty = tune(), mixture = tune()
) %>% set_engine("glmnet")

rf_reg <- rand_forest(
  mtry = tune(), trees = 1000, min_n = tune()
) %>% set_engine("ranger") %>% set_mode("regression")

xgb_reg <- boost_tree(
  trees = tune(), tree_depth = tune(), learn_rate = tune(),
  mtry = tune(), min_n = tune(), loss_reduction = tune()
) %>% set_engine("xgboost") %>% set_mode("regression")

# Recipe: trivial (already baked)
rec_reg_final <- recipe(bdi_post ~ ., data = X_reg) %>%
  update_role(id, new_role = "id") %>%
  step_zv(all_predictors())

# Grids
enet_grid <- grid_regular(
  penalty(range = c(-5, 1)),
  mixture(range = c(0, 1)),
  levels = 7
)
rf_grid_reg <- grid_random(
  mtry(range = c(5L, max(5L, floor(ncol(X_reg)/2)))),
  min_n(range = c(2L, 10L)),
  size = 20
)
xgb_grid_reg <- grid_random(
  trees(range = c(200L, 1500L)),
  tree_depth(range = c(2L, 6L)),
  learn_rate(range = c(0.01, 0.3)),
  mtry(range = c(5L, max(5L, floor(ncol(X_reg)/2)))),
  min_n(range = c(2L, 20L)),
  loss_reduction(),
  size = 30
)

# Workflows
wf_enet_reg <- workflow() %>% add_recipe(rec_reg_final) %>% add_model(enet_reg)
wf_rf_reg   <- workflow() %>% add_recipe(rec_reg_final) %>% add_model(rf_reg)
wf_xgb_reg  <- workflow() %>% add_recipe(rec_reg_final) %>% add_model(xgb_reg)

# Tuning (optimize RMSE)
ctrl_reg <- control_grid(save_pred = TRUE, save_workflow = TRUE)

res_enet_reg <- tune_grid(
  wf_enet_reg, resamples = folds_reg, grid = enet_grid,
  metrics = metric_set_reg, control = ctrl_reg
)
res_rf_reg <- tune_grid(
  wf_rf_reg, resamples = folds_reg, grid = rf_grid_reg,
  metrics = metric_set_reg, control = ctrl_reg
)
res_xgb_reg <- tune_grid(
  wf_xgb_reg, resamples = folds_reg, grid = xgb_grid_reg,
  metrics = metric_set_reg, control = ctrl_reg
)

best_enet_reg <- select_best(res_enet_reg, metric = "rmse")
best_rf_reg   <- select_best(res_rf_reg,   metric = "rmse")
best_xgb_reg  <- select_best(res_xgb_reg,  metric = "rmse")

# Save CV metrics
readr::write_csv(collect_metrics(res_enet_reg), "output/cv_reg_enet_metrics.csv")
readr::write_csv(collect_metrics(res_rf_reg),   "output/cv_reg_rf_metrics.csv")
readr::write_csv(collect_metrics(res_xgb_reg),  "output/cv_reg_xgb_metrics.csv")

# Final fits on full X_reg
final_enet_reg <- finalize_workflow(wf_enet_reg, best_enet_reg) %>% fit(X_reg)
final_rf_reg   <- finalize_workflow(wf_rf_reg,   best_rf_reg)   %>% fit(X_reg)
final_xgb_reg  <- finalize_workflow(wf_xgb_reg,  best_xgb_reg)  %>% fit(X_reg)

saveRDS(list(res = res_enet_reg, best = best_enet_reg, fit = final_enet_reg),
        "obj/final_enet_reg.rds")
saveRDS(list(res = res_rf_reg, best = best_rf_reg, fit = final_rf_reg),
        "obj/final_rf_reg.rds")
saveRDS(list(res = res_xgb_reg, best = best_xgb_reg, fit = final_xgb_reg),
        "obj/final_xgb_reg.rds")

# Importances
# ENet: coefficients
enet_coefs <- tidy(extract_fit_parsnip(final_enet_reg)) %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(estimate)))
readr::write_csv(enet_coefs, "output/reg_enet_coefficients.csv")

# RF: permutation importance
rf_reg_imp_spec <- rand_forest(
  mtry  = best_rf_reg$mtry,
  trees = 1000,
  min_n = best_rf_reg$min_n
) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("regression")

wf_rf_reg_imp <- workflow() %>%
  add_recipe(rec_reg_final) %>%
  add_model(rf_reg_imp_spec)

final_rf_reg_imp <- fit(wf_rf_reg_imp, X_reg)

# Now compute VI safely
rf_reg_vip <- vip::vi(extract_fit_parsnip(final_rf_reg_imp)$fit) %>%
  arrange(desc(Importance))
readr::write_csv(rf_reg_vip, "output/reg_rf_importance.csv")

p_rf_reg <- rf_reg_vip %>% slice_max(Importance, n = 20) %>%
  ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() + coord_flip() +
  labs(title = "Regression (All): Random Forest Permutation Importance",
       x = "Feature", y = "Importance")
ggsave("output/reg_rf_importance_top20.png", p_rf_reg, width = 9, height = 6, dpi = 300)

# XGB: gain importance
xgb_reg_fit <- extract_fit_parsnip(final_xgb_reg)$fit
xgb_reg_imp <- xgboost::xgb.importance(model = xgb_reg_fit)
readr::write_csv(as_tibble(xgb_reg_imp), "output/reg_xgb_importance.csv")
p_xgb_reg <- as_tibble(xgb_reg_imp) %>% slice_max(Gain, n = 20) %>%
  ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col() + coord_flip() +
  labs(title = "Regression (All): XGBoost Gain Importance",
       x = "Feature", y = "Gain")
ggsave("output/reg_xgb_importance_top20.png", p_xgb_reg, width = 9, height = 6, dpi = 300)

# Summaries
cv_best_reg <- bind_rows(
  collect_metrics(res_enet_reg) %>% mutate(model = "enet"),
  collect_metrics(res_rf_reg)   %>% mutate(model = "rf"),
  collect_metrics(res_xgb_reg)  %>% mutate(model = "xgb")
) %>%
  group_by(model, .metric) %>%
  summarise(mean = mean(mean), std_err = mean(std_err), .groups = "drop") %>%
  arrange(model, .metric)
readr::write_csv(cv_best_reg, "output/reg_cv_summary.csv")

message("\n=== Done ===")
message("• Classification CV summaries: output/cv_cls_* and output/cls_cv_summary.csv")
message("• Regression CV summaries:    output/cv_reg_* and output/reg_cv_summary.csv")
message("• Feature importance plots:   output/*_importance_top20.png")
message("• Coefficients tables:        output/*_coefficients.csv")


# ================================
# Extra plots the slides expect
# ================================

# 1) Classification – Elastic Net coefficients (top 20 by |estimate|)
#    Uses 'logit_coefs' already computed above
p_cls_enet <- logit_coefs %>%
  slice_max(abs(estimate), n = 20) %>%
  ggplot(aes(x = reorder(term, abs(estimate)), y = estimate)) +
  geom_col() +
  coord_flip() +
  labs(title = "Classification (MT): Elastic Net Coefficients (Top 20 by |β|)",
       x = "Feature", y = "Coefficient")
ggsave("output/cls_logit_coefficients_top20.png", p_cls_enet, width = 9, height = 6, dpi = 300)

# 2) Regression – Elastic Net coefficients (top 20 by |estimate|)
#    Uses 'enet_coefs' already computed above
p_reg_enet <- enet_coefs %>%
  slice_max(abs(estimate), n = 20) %>%
  ggplot(aes(x = reorder(term, abs(estimate)), y = estimate)) +
  geom_col() +
  coord_flip() +
  labs(title = "Regression (All): Elastic Net Coefficients (Top 20 by |β|)",
       x = "Feature", y = "Coefficient")
ggsave("output/reg_enet_coefficients_top20.png", p_reg_enet, width = 9, height = 6, dpi = 300)

# 3) Classification – model performance comparison (ROC AUC / PR AUC / Accuracy)
#    Uses 'cv_best_cls' already created above
p_cls_perf <- cv_best_cls %>%
  filter(.metric %in% c("roc_auc","pr_auc","accuracy")) %>%
  ggplot(aes(x = model, y = mean)) +
  geom_col() +
  facet_wrap(vars(.metric), scales = "free_y") +
  labs(title = "Classification (MT): Cross-validated performance by model",
       x = "Model", y = "Mean metric (5x5 CV)") +
  theme_minimal(base_size = 12)
ggsave("output/cls_model_performance.png", p_cls_perf, width = 10, height = 6, dpi = 300)

# 4) Regression – model performance comparison (RMSE / MAE / R²)
#    Uses 'cv_best_reg' already created above
p_reg_perf <- cv_best_reg %>%
  filter(.metric %in% c("rmse","mae","rsq")) %>%
  ggplot(aes(x = model, y = mean)) +
  geom_col() +
  facet_wrap(vars(.metric), scales = "free_y") +
  labs(title = "Regression (All): Cross-validated performance by model",
       x = "Model", y = "Mean metric (5x5 CV)") +
  theme_minimal(base_size = 12)
ggsave("output/reg_model_performance.png", p_reg_perf, width = 10, height = 6, dpi = 300)

# 5) Regression – Predicted vs Observed (best model: RF)
pred_reg_rf <- predict(final_rf_reg, X_reg) %>%
  bind_cols(truth = X_reg$bdi_post)

p_reg_scatter <- pred_reg_rf %>%
  ggplot(aes(x = truth, y = .pred)) +
  geom_point(alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = 2) +
  labs(title = "Regression (All): Predicted vs Observed (Random Forest)",
       x = "Observed BDI_post", y = "Predicted BDI_post")
ggsave("output/reg_pred_vs_obs_rf.png", p_reg_scatter, width = 7, height = 6, dpi = 300)

