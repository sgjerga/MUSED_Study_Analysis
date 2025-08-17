library("readr")
library("tidyverse")
library("janitor")
library("lubridate")
library("skimr")
library("glue")
library("tidyr")
library("dplyr")
library("tidyverse")

## ---- Paths & folders ----
dir.create("obj", showWarnings = FALSE)
dir.create("output", showWarnings = FALSE)

path_wide   <- "dataverse_files/MUSED_wide_2023_10-02.csv"
path_narrow <- "dataverse_files/MUSED_momentary_depression_2023_10-02.csv"
stopifnot(file.exists(path_wide), file.exists(path_narrow))

## ---- Load raw ----
wide_raw   <- readr::read_csv(path_wide,   show_col_types = FALSE)
narrow_raw <- readr::read_csv(path_narrow, show_col_types = FALSE)

## ---- Clean names & drop placeholder index ----
# clean_names() lowercases and snake_cases everything: BDI_pre -> bdi_pre
wide <- wide_raw |>
  janitor::clean_names() |>
  dplyr::select(-tidyselect::any_of(c("x1", "...1")))

narrow <- narrow_raw |>
  janitor::clean_names() |>
  dplyr::select(-tidyselect::any_of(c("x1", "...1")))

## ---- Basic type fixes / labels ----
# Assumption: group 1 = MusicTherapy, 0 = Control
wide <- wide |>
  dplyr::mutate(
    group        = as.integer(group),
    group_f      = factor(group, levels = c(0, 1), labels = c("Control", "MusicTherapy")),
    id           = as.integer(id),
    study_cohort = as.integer(study_cohort),
    dropout      = as.integer(dropout)
  )

# EMA file: time_pre_post 1=pre, 2=post
narrow <- narrow |>
  dplyr::mutate(
    id                = as.integer(id),
    group             = as.integer(group),
    study_cohort      = as.integer(study_cohort),
    time_pre_post_f   = factor(time_pre_post, levels = c(1, 2), labels = c("pre", "post")),
    time_day          = as.integer(time_day),
    time_moment       = as.integer(time_moment)
  )

## ---- Sanity check (optional) ----
# names(wide)[1:30]
# head(select(wide, bdi_pre, bdi_post, bdi_fu, hdrs_pre, hdrs_post))

## ---- Derive outcome deltas and responder flags ----
# NOTE: Use LOWERCASE variable names (bdi_pre, hdrs_post, ...) created by clean_names()
wide <- wide |>
  dplyr::mutate(
    # Absolute change (positive = improvement if higher score = worse)
    bdi_delta_post  = bdi_pre  - bdi_post,
    bdi_delta_fu    = bdi_pre  - bdi_fu,
    hdrs_delta_post = hdrs_pre - hdrs_post,
    
    # Percent change (guard against zero/NA baselines)
    bdi_pct_post  = dplyr::if_else(!is.na(bdi_pre)  & bdi_pre  > 0, (bdi_pre  - bdi_post)  / bdi_pre,  NA_real_),
    bdi_pct_fu    = dplyr::if_else(!is.na(bdi_pre)  & bdi_pre  > 0, (bdi_pre  - bdi_fu)    / bdi_pre,  NA_real_),
    hdrs_pct_post = dplyr::if_else(!is.na(hdrs_pre) & hdrs_pre > 0, (hdrs_pre - hdrs_post) / hdrs_pre, NA_real_),
    
    # Responder flags (>= 50% reduction)
    responder_bdi_post  = dplyr::if_else(!is.na(bdi_pct_post),  bdi_pct_post  >= 0.50, NA),
    responder_bdi_fu    = dplyr::if_else(!is.na(bdi_pct_fu),    bdi_pct_fu    >= 0.50, NA),
    responder_hdrs_post = dplyr::if_else(!is.na(hdrs_pct_post), hdrs_pct_post >= 0.50, NA)
  )

## ---- Done (up to the previous error point) ----
message("Clean names applied and deltas/responders computed successfully.")


## ---- Quick diagnostics (what EMA phases do we have?) ----
narrow |> count(time_pre_post, time_pre_post_f) |> arrange(time_pre_post)

## ---- Robust EMA summaries (handles missing pre/post) ----
library(tidyr)
library(dplyr)

mom_summary_raw <- narrow |>
  group_by(id, group, time_pre_post_f) |>
  summarise(
    mom_n      = n(),
    mom_mean   = mean(mom_depress, na.rm = TRUE),
    mom_median = median(mom_depress, na.rm = TRUE),
    mom_sd     = sd(mom_depress, na.rm = TRUE),
    .groups = "drop"
  )

# Ensure every id has both levels (pre & post). Missing ones become NA.
mom_summary_complete <- mom_summary_raw |>
  group_by(id, group) |>
  tidyr::complete(time_pre_post_f = factor(c("pre","post"), levels = c("pre","post"))) |>
  ungroup()

mom_summary <- mom_summary_complete |>
  tidyr::pivot_wider(
    names_from  = time_pre_post_f,
    values_from = c(mom_n, mom_mean, mom_median, mom_sd),
    names_glue  = "{.value}_{time_pre_post_f}",
    values_fill = NA
  ) |>
  mutate(
    mom_mean_delta_post   = mom_mean_pre   - mom_mean_post,
    mom_median_delta_post = mom_median_pre - mom_median_post
  )

## ---- Optional within-day slopes (also robust to missing phases) ----
mom_slopes_raw <- narrow |>
  group_by(id, time_pre_post_f, time_day) |>
  summarise(
    slope = if (sum(!is.na(mom_depress)) >= 3)
      tryCatch(coef(lm(mom_depress ~ time_moment))[2], error = function(e) NA_real_)
    else NA_real_,
    .groups = "drop"
  )

mom_slopes_complete <- mom_slopes_raw |>
  group_by(id, time_pre_post_f) |>
  summarise(slope_mean = mean(slope, na.rm = TRUE), .groups = "drop") |>
  group_by(id) |>
  tidyr::complete(time_pre_post_f = factor(c("pre","post"), levels = c("pre","post"))) |>
  ungroup()

mom_slopes <- mom_slopes_complete |>
  tidyr::pivot_wider(
    names_from  = time_pre_post_f,
    values_from = slope_mean,
    names_glue  = "slope_mean_{time_pre_post_f}",
    values_fill = NA
  ) |>
  mutate(slope_mean_delta_post = slope_mean_pre - slope_mean_post)

## ---- Combine EMA summaries and continue as before ----
mom_summary_full <- mom_summary |>
  left_join(mom_slopes, by = "id")

wide_plus <- wide |>
  left_join(mom_summary_full, by = "id")


## ---- Long format for repeated outcomes (BDI & HDRS) ----
time_map <- tibble::tibble(
  timepoint = c("pre","post","fu"),
  time_num  = c(0, 1, 2)
)

wide_long <- wide |>
  dplyr::select(id, group_f, dplyr::starts_with("bdi_"), dplyr::starts_with("hdrs_")) |>
  tidyr::pivot_longer(
    cols = -c(id, group_f),
    names_to   = c(".value", "timepoint"),
    names_sep  = "_"
  ) |>
  dplyr::left_join(time_map, by = "timepoint") |>
  dplyr::arrange(id, time_num)

## ---- Save processed objects ----
readr::write_rds(narrow,           "obj/mused_momentary_clean.rds")
readr::write_rds(mom_summary_full, "obj/mused_momentary_summary.rds")
readr::write_rds(wide_plus,        "obj/mused_wide_plus_momentary.rds")
readr::write_rds(wide_long,        "obj/mused_wide_long.rds")

# (Optional) CSVs
readr::write_csv(mom_summary_full, "obj/mused_momentary_summary.csv")
readr::write_csv(wide_long,        "obj/mused_wide_long.csv")

## ---- QC plots ----
theme_set(ggplot2::theme_minimal(base_size = 12))

# 1) BDI trajectories by group
p1 <- wide_long |>
  dplyr::filter(!is.na(bdi)) |>
  ggplot2::ggplot(ggplot2::aes(x = timepoint, y = bdi, group = id, color = group_f)) +
  ggplot2::geom_line(alpha = 0.25) +
  ggplot2::stat_summary(
    ggplot2::aes(group = group_f),
    fun = stats::median, geom = "line", linewidth = 1.3
  ) +
  ggplot2::labs(title = "BDI over time by group",
                subtitle = "Light lines = individuals; bold = median",
                x = "Time", y = "BDI", color = "Group")
ggplot2::ggsave("output/bdi_trajectories.png", p1, width = 7, height = 5, dpi = 300)

wide_long |> group_by(group_f, timepoint) |> summarise(median_bdi = median(bdi, na.rm=TRUE))

# 2) ΔBDI (pre → post) by group
p2 <- wide |>
  dplyr::filter(!is.na(bdi_delta_post)) |>
  ggplot2::ggplot(ggplot2::aes(x = group_f, y = bdi_delta_post, fill = group_f)) +
  ggplot2::geom_violin(trim = FALSE, alpha = 0.3) +
  ggplot2::geom_boxplot(width = 0.15, outlier.shape = NA) +
  ggplot2::geom_hline(yintercept = 0, linetype = 2) +
  ggplot2::labs(title = "Change in BDI (pre → post) by group",
                y = "BDI improvement (positive = improvement)", x = NULL) +
  ggplot2::theme(legend.position = "none")
ggplot2::ggsave("output/bdi_change_by_group.png", p2, width = 6, height = 4.5, dpi = 300)

wide |> group_by(group_f) |> summarise(
  median_delta = median(bdi_delta_post, na.rm=TRUE),
  mean_delta   = mean(bdi_delta_post, na.rm=TRUE)
)


# 3) EMA: since there's no post, show pre-only distribution (+ by group)
narrow_pre <- narrow |>
  filter(time_pre_post_f == "pre") |>
  left_join(dplyr::select(wide, id, group_f), by = "id")

# Plot 3: PRE EMA histogram by randomized group (robust)
p3 <- narrow_pre |>
  ggplot2::ggplot(ggplot2::aes(x = mom_depress, fill = group_f)) +
  ggplot2::geom_histogram(bins = 30, alpha = 0.5, position = "identity", na.rm = TRUE) +
  ggplot2::labs(title = "Momentary depression at PRE (EMA)",
                x = "Momentary depression", y = "Count", fill = "Group")
ggplot2::ggsave("output/ema_pre_distribution.png", p3, width = 6, height = 4.5, dpi = 300)

# Numbers for Plot 3: baseline EMA by group
narrow_pre |> 
  group_by(group_f) |> 
  summarise(
    n_obs       = sum(!is.na(mom_depress)),
    mean_pre_ema   = mean(mom_depress, na.rm = TRUE),
    median_pre_ema = median(mom_depress, na.rm = TRUE)
  )


# 4) Optional: EMA per-person PRE means by group (box/violin)
ema_pre_means <- narrow |>
  dplyr::filter(time_pre_post_f == "pre") |>
  dplyr::group_by(id) |>
  dplyr::summarise(mom_mean_pre = mean(mom_depress, na.rm = TRUE), .groups = "drop") |>
  dplyr::left_join(dplyr::select(wide, id, group_f), by = "id")

p4 <- ema_pre_means |>
  ggplot2::ggplot(ggplot2::aes(x = group_f, y = mom_mean_pre, fill = group_f)) +
  ggplot2::geom_violin(trim = FALSE, alpha = 0.3) +
  ggplot2::geom_boxplot(width = 0.15, outlier.shape = NA) +
  ggplot2::labs(title = "EMA PRE (per-person means) by randomized group",
                x = NULL, y = "Mean momentary depression (pre)") +
  ggplot2::theme(legend.position = "none")
ggplot2::ggsave("output/ema_pre_means_by_group.png", p4, width = 6, height = 4.5, dpi = 300)

ema_pre_means |> group_by(group_f) |>
  summarise(
    median_pre_mean = median(mom_mean_pre, na.rm=TRUE),
    mean_pre_mean = mean(mom_mean_pre, na.rm=TRUE)
  )


## Check Missing Data Patterns
library(naniar)

# Missingness summary
miss_summary <- wide_plus |> 
  summarise(across(everything(), ~ sum(is.na(.))))
print(miss_summary)

# Visual missingness pattern
naniar::vis_miss(wide_plus)
ggplot2::ggsave("output/missing_patterns.png", naniar::vis_miss(wide_plus), width = 12, height = 5, dpi = 300)

# Missingness by group
wide_plus |> 
  group_by(group_f) |> 
  summarise(across(everything(), ~ mean(is.na(.))))


## Baseline Comparisons Across Many Variables
baseline_vars <- c("bdi_pre", "hdrs_pre", "mom_mean_pre", "age")

baseline_stats <- wide_plus |> 
  summarise(across(all_of(baseline_vars),
                   list(mean = ~ mean(., na.rm = TRUE),
                        median = ~ median(., na.rm = TRUE),
                        sd = ~ sd(., na.rm = TRUE))))

print(baseline_stats)

# Group comparison
wide_plus |> 
  group_by(group_f) |> 
  summarise(across(all_of(baseline_vars),
                   list(mean = ~ mean(., na.rm = TRUE),
                        median = ~ median(., na.rm = TRUE),
                        sd = ~ sd(., na.rm = TRUE))))


## Correlation Matrix
library(GGally)

corr_data <- wide_plus |> 
  dplyr::select(bdi_pre, hdrs_pre, mom_mean_pre, bdi_post, hdrs_post, mom_mean_post)

# GGally::ggpairs(corr_data)


## Check Responder Profiles
wide_plus |> 
  group_by(group_f) |> 
  summarise(
    bdi_resp_rate = mean(responder_bdi_post, na.rm = TRUE),
    hdrs_resp_rate = mean(responder_hdrs_post, na.rm = TRUE)
  )


## Time Trends in EMA
narrow |> 
  filter(time_pre_post_f == "pre") |> 
  group_by(time_day, group) |> 
  summarise(mean_mom = mean(mom_depress, na.rm = TRUE)) |> 
  ggplot(aes(x = time_day, y = mean_mom, color = factor(group))) +
  geom_line() +
  labs(title = "EMA Trend over Days (Pre)", color = "Group")


## Check Cohort Effects
wide_plus |> 
  group_by(study_cohort, group_f) |> 
  summarise(mean_bdi_pre = mean(bdi_pre, na.rm = TRUE))


## Distribution checks for all cohorts of the data
delta_vars <- c("bdi_delta_post", "bdi_delta_fu", "hdrs_delta_post")

wide_plus |> 
  dplyr::select(all_of(delta_vars)) |> 
  summary()

