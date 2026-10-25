| key | parent | value | real_or_assumed | justification |
| --- | --- | --- | --- | --- |
| enabled | layers | false | design | n/a |
| n_partitions | n/a | 512 | real | n/a |
| n_nodes | n/a | 16 | real | n/a |
| headroom | n/a | 1.6 | assumed | n/a |
| slo_threshold | n/a | 1 | design | n/a |
| window_steps | n/a | 720 | design | n/a |
| step_seconds | n/a | 60 | real | n/a |
| migration_steps_m0 | n/a | 5 | assumed | n/a |
| migration_overhead_kappa | n/a | 0.3 | assumed | n/a |
| migration_budget_B | n/a | 4 | design | n/a |
| monitor_delay_d | n/a | 1 | design | n/a |
| decision_interval_k | n/a | 1 | design | n/a |
| horizons | n/a | [1, 2, 5, 10] | design | n/a |
| refit_interval | n/a | 360 | assumed | n/a |
| seasonal_period | n/a | 1440 | real | n/a |
| tuning_budget | n/a | 36 | design | n/a |
| tune_plateau_tol | n/a | 0.05 | assumed | n/a |
| ewma_alpha_grid | n/a | [0.1, 0.3, 0.5, 0.8] | design | n/a |
| theta_hi_grid | n/a | [0.8, 0.9, 1.0] | design | n/a |
| cooldown_grid | n/a | [0, 5, 15] | design | n/a |
| hysteresis_grid | n/a | [0.0, 0.1, 0.2, 0.5] | design | n/a |
| oracle_window_grid | n/a | [1, 5, 10] | design | n/a |
| margin_quantile | n/a | 0.9 | design | n/a |
| replicates_per_cell | n/a | 30 | design | n/a |
| min_replicates | n/a | 30 | design | n/a |
| m_levels | n/a | [1, 2, 5, 10] | design | n/a |
| headroom_levels | n/a | [1.3, 1.6, 2.0] | design | n/a |
| pilot_replicates | n/a | 5 | design | n/a |
| oracle_check_events | n/a | 100 | design | n/a |
| n_boot | n/a | 9999 | design | n/a |
| ci_level | n/a | 0.95 | design | n/a |
| r | n/a | 5 | design | n/a |
| W_b | n/a | 60 | design | n/a |
| s | n/a | 3 | design | n/a |
| eps | n/a | 0.001 | design | n/a |
| rise_lo | n/a | 0.1 | design | n/a |
| rise_hi | n/a | 0.9 | design | n/a |
| simplex_step | n/a | 0.1 | design | n/a |
| min_events | n/a | 24 | design | n/a |
| min_lag_tax_share | n/a | 0.05 | design | n/a |
| capacity_jitter_frac | n/a | 0.1 | assumed | n/a |
| wasted_move_window | n/a | 10 | design | n/a |
| inject_tolerance | n/a | 5 | design | n/a |
| injection_amplitude_min | n/a | 10 | assumed | n/a |
| injection_amplitude_max | n/a | 100 | assumed | n/a |
| injection_rise_min | n/a | 2 | assumed | n/a |
| injection_rise_max | n/a | 10 | assumed | n/a |
| injection_dur_min | n/a | 10 | assumed | n/a |
| injection_dur_max | n/a | 50 | assumed | n/a |
| cpu_cores | n/a | 8 | real | n/a |
| h_net | n/a | 2 | assumed | n/a |
| rho | n/a | 0.25 | assumed | n/a |
| sigma_size | n/a | 0.5 | assumed | n/a |
| phi | n/a | 1 | assumed | n/a |
| master_seed | n/a | 42 | design | n/a |
| hash_seed | n/a | 'seed1' | design | n/a |
| reserved_cores | n/a | 1 | design | n/a |
| workers | n/a | 7 | design | n/a |
| compute_budget_core_hours | n/a | 288 | real | n/a |
| runner_stall_minutes | n/a | 30 | design | n/a |
| cleanroom_stride | n/a | 1 | design | n/a |
| d2_max_gb | n/a | 2 | real | n/a |
| azure_days | n/a | 7 | real | n/a |
| azure_skip_cols | n/a | 4 | real | n/a |
| d2_min_cols | n/a | 4 | real | n/a |
| c0 | n/a | 1 | assumed | n/a |
| c1 | n/a | 0 | assumed | n/a |
| azure_mins_per_day | n/a | 1440 | real | n/a |
| data_root | n/a | 'data' | real | n/a |
| ridge_alpha | n/a | [0.1, 1.0, 10.0] | design | n/a |
| lightgbm_leaves | n/a | [15, 31, 63] | design | n/a |
| gamma | n/a | 0.005 | design | ACI step size, the value used in the A15 experiments (Gibbs and Candes 2021, Sec 2.1), not tuned here (A26) |
| a_min | n/a | 0.01 | assumed | n/a |
| a_max | n/a | 0.2 | assumed | n/a |
| residual_window | n/a | 100 | design | n/a |
| psi_grid | n/a | [0.1, 0.2, 0.5] | design | n/a |
| io_chunk_bytes | compute | 65536 | design | n/a |
| snapshots | paths | 'snapshots' | design | n/a |
| default_tag | snapshot | 'prereg-v1' | design | n/a |
| tag_pattern | snapshot | '[a-zA-Z0-9_-]+' | design | n/a |
| sources | snapshot | ['src', 'scripts', 'configs'] | design | n/a |
| ignore_patterns | snapshot | ['*.pyc', '__pycache__'] | design | n/a |
| manifest_name | snapshot | 'manifest.json' | design | n/a |
| raw_dirname | n/a | 'raw' | design | n/a |
| azure_dirname | n/a | 'azure2019' | design | n/a |
| twitter_dirname | n/a | 'twitter' | design | n/a |
| processed_dirname | n/a | 'processed' | design | n/a |
| logs_dirname | n/a | 'logs' | design | n/a |
| amendments_file | n/a | 'AMENDMENTS.md' | design | n/a |
| d2_file | n/a | 'cluster53.sort.zst' | real | n/a |
| azure_invocation_pattern | n/a | 'invocations_per_function_md.anon.d{day:02d}.csv' | real | n/a |
| azure_duration_pattern | n/a | 'function_durations_percentiles.anon.d{day:02d}.csv' | real | n/a |
| azure_memory_pattern | n/a | 'app_memory_percentiles.anon.d{day:02d}.csv' | real | n/a |
| azure_head_cols | n/a | ['HashOwner', 'HashApp', 'HashFunction', 'Trigger'] | real | n/a |
| azure_id_cols | n/a | ['HashOwner', 'HashApp', 'HashFunction'] | real | n/a |
| azure_dur_avg_col | n/a | 'Average' | real | n/a |
| azure_dur_count_col | n/a | 'Count' | real | n/a |
| azure_memory_cols | n/a | ['HashOwner', 'HashApp', 'SampleCount', 'AverageAllocatedMb', 'AverageAllocatedMb_pct1', 'AverageAllocatedMb_pct5', 'AverageAllocatedMb_pct25', 'AverageAllocatedMb_pct50', 'AverageAllocatedMb_pct75', 'AverageAllocatedMb_pct95', 'AverageAllocatedMb_pct99', 'AverageAllocatedMb_pct100'] | real | n/a |
| azure_chunk_rows | n/a | 5000 | design | n/a |
| d2_chunk_rows | n/a | 2000000 | design | n/a |
| d2_cols | n/a | ['timestamp', 'key', 'key_size', 'value_size', 'client', 'op', 'ttl'] | real | n/a |
| d2_time_col | n/a | 'timestamp' | real | n/a |
| d2_key_col | n/a | 'key' | real | n/a |
| d2_ksize_col | n/a | 'key_size' | real | n/a |
| d2_vsize_col | n/a | 'value_size' | real | n/a |
| d2_op_col | n/a | 'op' | real | n/a |
| d2_step_seconds | n/a | 10 | design | n/a |
| d1_train_days | n/a | 2 | design | n/a |
| d1_tune_days | n/a | 1 | design | n/a |
| d2_split_fractions | n/a | [0.25, 0.15] | design | n/a |
| key_sep | n/a | '\|' | design | n/a |
| hash_bits | n/a | 128 | real | n/a |
| d1_bytes_sigma | n/a | 0.5 | assumed | n/a |
| d1_bytes_label | n/a | 'bytes' | design | n/a |
| demand_fallback | n/a | 1 | assumed | n/a |
| inject_label | n/a | 'D3' | design | n/a |
| injection_events | n/a | 40 | design | n/a |
| injection_hot_min | n/a | 1 | assumed | n/a |
| injection_hot_max | n/a | 4 | assumed | n/a |
| wait_quantile | n/a | 0.99 | design | n/a |
| tune_replicates | n/a | 35 | design | n/a |
| tune_plateau_frac | n/a | 0.25 | design | n/a |
| seed_bits | n/a | 32 | design | n/a |
| day_seconds | n/a | 86400 | real | n/a |
| lookback_L | n/a | 30 | design | n/a |
| roll_window | n/a | 10 | design | n/a |
| forecast_ewma_alpha | n/a | 0.3 | design | n/a |
| origin_stride | n/a | 4 | design | n/a |
| lgbm_trees | n/a | 100 | design | n/a |
| lgbm_lr | n/a | 0.1 | design | n/a |
| lgbm_verbosity | n/a | -1 | design | n/a |
| feature_chunk_origins | n/a | 256 | design | n/a |
| forecaster_names | n/a | ['naive', 'ewma', 'seasonal_naive', 'ridge', 'lightgbm'] | design | n/a |
| warn_ratio | n/a | 3 | design | n/a |
| warn_baseline_window | n/a | 60 | design | n/a |
| smape_eps | n/a | 0.001 | design | n/a |
| forecast_dirname | n/a | 'forecasts' | design | n/a |
| h_net_levels | n/a | [1.3, 2.0, 4.0] | assumed | n/a |
| phi_levels | n/a | [1.0, 0.1, 0.01] | assumed | n/a |
| size_label | n/a | 'part_size' | design | n/a |
| budget_levels | n/a | [1, 2, 8, 'unlimited'] | design | AA5 levels besides the default 4; unlimited means n_partitions (A34) |
| margin_levels | n/a | [0.8, 0.9, 0.95] | design | AA6 margin levels for P4 and P4a (A34) |
| lead_levels | n/a | [0, 1] | design | AA2 steps ahead of t the policy targets besides the default m: current step 0, next step 1 (A34) |
| rise_buckets | n/a | 5 | design | D3 rise-time buckets of equal event count (A36) |
| D1 | q2_forecast_variants | P3_naive | design | 14.9053 against P4 15.8977 and P3_ewma 15.5024 |
| D2 | q2_forecast_variants | P3_ewma | design | 0.00285714 against P3_naive 0.00480893 and P4 0.0683602 |
| interaction_effect_frac | n/a | 0.05 | design | minimum effect size, as a fraction of the default mean damage, for a Q1 factor to count as active in the interaction trigger (A38) |
| interaction_min_active | n/a | 2 | design | the interaction trigger fires for a dataset and policy when at least this many factors are active (A38) |
| skew_hot_share | n/a | 0.25 | design | RV2 skewed placement: share of partitions, heaviest by training-prefix mean load, packed on the hot nodes (A38) |
| skew_hot_nodes | n/a | 4 | design | RV2 skewed placement: number of hot nodes, the first nodes in index order (A38) |
| kappa_levels | n/a | [0.2, 0.3, 0.5] | assumed | AA10 migration overhead levels (guide section 3); the default 0.3 rows come from the Must grid and Q1 AA6 (A41) |
| d3_learned_forecasters | n/a | ['lightgbm'] | design | learned forecasters whose D3 tables are built for RV0, tuned hyperparameter of the source dataset (A42) |
| throttle_budget_B | n/a | 1 | design | RV3 throttled policy: the lowest AA5 budget level, against the default migration_budget_B (A45) |
| monitor_delay_levels | n/a | [0, 1, 3] | design | AA7 monitoring delay d, the guide config table; the default 1 is skipped in Q6 (A49) |
| decision_interval_levels | n/a | [1, 5, 15] | design | AA8 decision interval k, the guide config table; the default 1 is skipped in Q6 (A49) |
| chronos_model | n/a | 'amazon/chronos-bolt-small' | design | n/a |
| chronos_context | n/a | 128 | design | n/a |
| chronos_stride | n/a | 4 | design | n/a |
| chronos_quantile | n/a | 0.5 | design | n/a |
| chronos_batch | n/a | 512 | design | n/a |
| h2_split | n/a | 0.5 | design | H2 split on the median recovery ratio R between abrupt and ramped onsets (guide Section 3 config table; A56) |
| h3_ub | n/a | 0.2 | design | H3 upper bound on median R at m = 1 (guide Section 3 config table; A56) |
| h4_rho | n/a | 0.5 | design | H4 bound on the absolute Spearman correlation of MASE with placement benefit (guide Section 3 config table; A56) |
| h4_ub | n/a | 0.7 | design | H4 bound on the upper end of the BCa interval of that correlation (guide Section 3 config table; A56) |
| h5_ratio | n/a | 1.25 | design | H5 minimum transfer ratio, cross-dataset over in-dataset (guide Section 3 config table; A56) |
| h6_tau | n/a | 0.6 | design | H6 minimum Kendall tau of the D1 and D2 policy rankings (guide Section 3 config table; A56) |
| h9_share | n/a | 0.05 | design | H9 minimum share of violating node-steps attributed to the network or migration layer (guide Section 3 config table; A56) |
| alpha_adj | n/a | 0.05 | design | significance level after Holm correction, every family (guide Section 3 config table; A56) |
| h12_noninf | n/a | 0.02 | design | H12 non-inferiority margin for the damage of P4a against P4 (guide Section 3 config table; A56) |
| must | result_folders | 'results' | design | n/a |
| q1 | result_folders | 'results/q1' | design | n/a |
| q2 | result_folders | 'results/q2' | design | n/a |
| q3 | result_folders | 'results/q3' | design | n/a |
| q4 | result_folders | 'results/q4' | design | n/a |
| q6 | result_folders | 'results/q6' | design | n/a |
| q8 | result_folders | 'results/q8' | design | n/a |
| q11_rv0 | result_folders | 'results/q11_rv0' | design | n/a |
| q11_rv123 | result_folders | 'results/q11_rv123' | design | n/a |
