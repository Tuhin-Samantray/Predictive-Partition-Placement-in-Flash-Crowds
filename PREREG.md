# PREREG.md: Project A pre-registration

Date: 2026-10-23. Written before the simulator, any forecaster or any policy exists; no policy has been run.
Sources: Project_A_Execution_Guide_LLM.docx (dated 2026-10-19) and Tuhin_Review_comments.docx, both read 2026-10-23.
Lock: the SHA-256 of this file is stored in PREREG.sha256 beside it, never inside it. Pre-registered parameter values are those of configs/default.yaml as copied into snapshot prereg-v1 (see its manifest.json). After the lock this file is not edited. Every change is a dated entry in AMENDMENTS.md stating whether it was made before any policy run.
Tags: real = measured or cited; assumed = stated modelling assumption; design = design value.

## 1. Scope decisions made by the author before the lock

1. Undefined in the source guide, so not run and not claimed: AA12, FA1, H7, H8 (numbering of the others is kept). They can enter only by a dated amendment stating the source definition, made before the first policy run on evaluation data. Anything added later is labelled exploratory.
2. The Azure application-memory files are downloaded and logged but used by no computation. Partition size S_p comes from the assumed model in section 5. Using the files requires a dated amendment.
3. Out of scope, future work: replication layer L4, CL7, geography L5, CL8, fractional factorial designs, replication lag.
4. The extensions requested in Tuhin_Review_comments.docx are pre-registered in section 9, so none is labelled pre-registered after the fact. Auditing the earlier simulator (its 70 runs, manifest and results) is separate work. Nothing from it sets a parameter, hypothesis or threshold here, and no earlier result is reused.
5. Naming: policy IDs follow the guide (P0 to P6, P4a). The review's R0, R1, R2 are P0, P1, P2. Generated artifacts keep the guide's IDs (TA1 to TA9, FA2 to FA16). The review's tables and figures are generated as RT1 to RT11 and RF1 to RF10 and mapped in Phase 14.
6. Pilot runs (Phase 7) and all tuning use the tuning split only. No policy is run on evaluation data before the grid is launched from a snapshot.

## 2. Research questions

- RQ1: how much flash-crowd damage is due to reactive lag (lag tax = reactive minus oracle)?
- RQ2: what fraction of the lag tax do realistic forecasters recover (recovery ratio R)?
- RQ3: how does R depend on onset speed and on migration duration?
- RQ4: what does proactivity cost in extra and wasted moves?
- RQ5: does forecast accuracy (MASE) predict placement benefit?
- RQ6: does the answer change when CPU and network layers are modelled?
- RQ7: does it change under a disaggregated-storage migration model (a move transfers a fraction phi of the partition)?

## 3. Hypotheses (thresholds are keys in configs/default.yaml, section thresholds)

| ID | Prediction | If wrong, the paper says |
| --- | --- | --- |
| H1 | Lag tax is positive and grows with migration duration m | Reactive placement is already near-optimal at the tested scales |
| H2 | Median R is below h2_split for abrupt onsets and above it for ramped onsets | Forecast usefulness does not depend on onset shape |
| H3 | R falls toward 0 when m is 1 step or less (upper bound h3_ub) | Benefit persists with cheap migration |
| H4 | Spearman correlation of forecaster MASE with placement benefit is below h4_rho in magnitude | Accuracy is a good proxy for benefit |
| H5 | Tuned parameters and forecasters transfer poorly across D1 and D2 (ratio at least h5_ratio) | Gains transfer across datasets |
| H6 | Policy ranking is stable across datasets (Kendall tau at least h6_tau) while the size of R differs | Best-policy conclusions are dataset-specific |
| H9 | With tight bandwidth, a CPU-only forecast policy (P3) leaves a share of SLO violations above h9_share attributable to network or migration interference | CPU-only policies suffice |
| H10 | Congestion-coupled migration lowers R relative to constant-m | Coupling does not change R |
| H11 | Under the disaggregated scenario (phi well below 1) the lag tax and R shrink, consistent with H3 | Migration cost is not the main limit on forecast benefit |
| H12 | The adaptive-conformal margin (P4a) causes fewer wasted moves than the fixed-quantile margin (P4) at equal or lower damage (non-inferiority h12_noninf) | No difference, or more wasted moves |

All confirmatory comparisons are in section 7. Anything else is exploratory.

## 4. Data, splits, events

- D1: Azure Functions Trace 2019, days 1 to 7 (invocation counts; duration files for X1). Invocations are a demand proxy, not database queries.
- D2: one Twitter cache cluster. Selection rule: among clusters whose compressed file is under d2_max_gb (2 GiB = 2147483648 bytes, the unit of the server listing), take the largest. Result: cluster53, 2084971699 bytes. If Phase 3 finds fewer than min_events events, take the next largest under the cap in the order 45, 50, 26, 10, logging every attempt with detector counts only, never policy results.
- D3: injected spikes on D1 and D2 baselines (amplitude, rise time, duration, key concentration from seeded config distributions).
- D4: Azure 2021 trace, conditional (Phase 11), only if download and parsing succeed.
- Mapping: function or key to one of P partitions by MD5 modulo P with hash_seed. L[t, p] at step_seconds (D1 60, D2 10). The simulator never applies log1p; forecasters do internally.
- Splits, fixed now: D1 days 1 to 2 train forecasters, day 3 is the tuning split, days 4 to 7 are evaluation. D2: first 25 percent of time trains, next 15 percent tunes, the remainder is evaluation. Capacity comes from the training prefix only.
- Replicate: a (window, hash seed, capacity-jitter seed) triple whose window contains one detected onset at a random offset. The independent unit is the event: intervals are clustered by event id and distinct events per dataset are reported. At least min_events events per dataset are required, otherwise detector r or s is lowered by amendment before any policy run, or the regime map relies on D3.
- Detector: event when load is at least r times its trailing median over the baseline window (ending at t minus 1) and stays so for the sustain steps. Output: onset, peak, rise time (rise_lo to rise_hi), duration, partition, event id.
- Onset class: ramped if rise time is at or above the dataset median rise time, abrupt otherwise. The cut point is computed once in Phase 3 and recorded as a dated amendment before any policy run.

## 5. Simulator definitions

- P partitions on N nodes. A policy at step t sees L only up to t minus d and decides every k steps. A move started at t completes at t + m; the source serves the partition and pays factor (1 + kappa). Reactive lag = d + k + m.
- Capacity per node mu = h x (mean total load per step in the training prefix) / N. Fluid queue Q(t+1) = max(0, Q(t) + A(t) - mu); wait W = (Q + A/2) / mu. Damage D = sum over nodes and steps of max(0, W - S) x A, normalized by total arrivals.
- Lag tax = D_reactive - D_oracle (the oracle uses the same mechanism with true load at t + m). R = (D_reactive - D_forecast) / (D_reactive - D_oracle), with D_reactive the best tuned reactive policy. R is reported only when the lag tax is at least min_lag_tax_share of D_reactive. The share of replicates with R defined is always reported.
- Per-replicate metrics: D, SLO-violation fraction, p99 wait proxy, overload steps, peak backlog, imbalance (max over mean utilization), moves, migrated load and bytes, wasted moves (the source node would not have exceeded capacity within wasted_move_window steps under true load).
- Cross-layer model (X1, X2; every conclusion is "under the stated model"): CPU layer u_cpu = sum of load x service demand over the partitions on a node, divided by cores per step. NIC layer: foreground bytes per node-step (D2 real from key plus value size; D1 assumed lognormal); link capacity = h_net x mean foreground bytes per node-step on the training prefix. Migration is a flow of phi x S_p bytes at rate min(rho x link, free residual capacity divided by in-flight moves at that node), charged to source and destination links, with a floor of one step. Partition size S_p = m0 x rho x link bytes per step x z_p, with z_p lognormal of mean 1 and sigma sigma_size. Wait = max(CPU wait, NIC wait); the binding layer is recorded for every violating node-step. Constant-m mode is the control. Metrics CL1 to CL6 and CL9 as in the guide.
- Tiers: h_net tight 1.3, moderate 2.0, abundant 4.0 (relative to trace load, no hardcoded Gbps).

## 6. Policies and tuning

- P0 static hash. P1 reactive threshold with cooldown (EWMA, tuned). P2 reactive greedy repack every k steps (EWMA, tuned). P3 forecast-driven at completion time t + m with hysteresis delta, one variant per forecaster. P4 is P3 with a rolling upper-quantile margin. P4a is P4 with an adaptive-conformal margin (A15). P5 oracle (true load at t + m). P6 cross-layer forecast (max of predicted CPU and NIC utilization; delays a move when free bandwidth is below psi). Same budget, delay and overhead for all; only information differs.
- Forecasters: naive, EWMA, seasonal-naive, Ridge, LightGBM (global, one model per horizon, refit every refit_interval, rolling origin). Forecasts are precomputed once as F[t, h, p] (float32) per (dataset, forecaster). Chronos (X5) is accuracy only.
- Tuning: seeded ParameterSampler over the config grids, configs_per_policy configurations and at least min_replicates replicates each on the tuning split, identical budget for every tunable policy. If the best-so-far gain over the last plateau_window iterations exceeds plateau_tol, the budget is raised equally for all tunables by amendment. P4 and P4a use the forecaster with the lowest tuning-split MASE. The best tuned reactive policy (lower tuning-split damage of P1 and P2) is written to tuned.yaml before the grid.

## 7. Statistics plan

Comparisons are paired by replicate against the best tuned reactive policy. Intervals: BCa bootstrap with n_boot resamples, clustered by event. Holm correction within each family. Effect sizes (Cliff's delta, or difference in median R) accompany every p-value. The total number of tests is reported. Labels assigned by the analysis script: supported, not supported (interval or test points the other way or misses the threshold with adequate sample), inconclusive (interval spans the threshold, or fewer than min_replicates or min_events).

| ID | Hyp | Quantity and test | Support rule | Holm family |
| --- | --- | --- | --- | --- |
| C1 | H1 | Lag tax per replicate: one-sided Wilcoxon signed-rank against 0; Spearman of tax against m | Tax above 0, adjusted p below alpha_adj on both datasets, rho above 0 | F1: 2 datasets x 2 tests |
| C2 | H2 | Median R by onset class with BCa; Mann-Whitney U ramped vs abrupt; Cliff's delta | Abrupt below h2_split, ramped above, intervals exclude it, U significant | F2: 2 datasets |
| C3 | H3 | Median R at m = 1 with BCa; Mann-Whitney U against m at least 5 | Upper bound below h3_ub | F3: 2 datasets |
| C4 | H4 | Spearman of MASE vs placement benefit over (forecaster x dataset x onset class) cells, with BCa | Absolute rho below h4_rho and interval upper bound below h4_ub; state low power | single test |
| C5 | H5 | MASE transfer ratio (fit on one dataset, evaluate on the other) over in-dataset MASE; XA2 tuning regret | Median ratio at least h5_ratio | F4: forecasters |
| C6 | H6 | Kendall tau of D1 and D2 policy rankings by median damage | Tau at least h6_tau | single test |
| C7 | H9 | Share of violating node-steps attributed to NIC or migration under P3 at the tight tier (CL3); paired Wilcoxon of damage P3 vs P6 | Share interval lower bound above h9_share; P6 damage lower, adjusted p below alpha_adj | F5: datasets x tiers |
| C8 | H10 | Per-replicate R(coupled) minus R(constant m): Wilcoxon signed-rank | Difference below 0, adjusted p below alpha_adj | F6: datasets x tiers |
| C9 | H11 | Lag tax and R at phi 0.01 vs 1.0: paired Wilcoxon | Both smaller, adjusted p below alpha_adj | F7: datasets x 2 metrics |
| C10 | H12 | P4a vs P4: wasted moves (one-sided, lower); damage non-inferiority | Both hold (margin h12_noninf) | F8: datasets x 2 metrics |
| C11 | Main | Each of P0, P2, P3, P4, P4a, P5, P6 vs best tuned reactive: Wilcoxon signed-rank, Cliff's delta | Reported in full | F9: policies x datasets x onset class |
| C12 | Ablations, robustness | Each level vs default: effect size with BCa, Wilcoxon; covers AA1 to AA13 (except AA12), XA6, FA16 | Reported in full | one family per ablation |
| C13 | Review | P4a vs each of P1, P0, P3, P4, P5 (P3 uses the forecaster chosen for P4): paired Wilcoxon on damage per replicate, mean paired difference with BCa interval | Reported in full | F10: 5 pairs x datasets |

## 8. Runs

- Must grid: P0, P1, P2, P5, P3 x 5 forecasters, P4 variants, x m in {1, 2, 5, 10} x headroom in {1.3, 1.6, 2.0} x D1, D2 x onset class, 30 replicates per cell (about 480 cells, 14,400 runs; confirmed by the dry-run).
- Q1 ablations AA1 to AA6 and AA9, one factor at a time around default (m 5, h 1.6), per dataset. Q2 D3 regime map: injected rise-time buckets x m for P1, the best forecast policy and P5. Q3 cross-layer: P1, P3, P6, P5 x h_net tiers x mode (coupled, constant m) x phi x dataset. Q4 AA10, AA11 (P3 vs P6 with cross-layer on and off), P4 vs P4a. Q5 forecast-only cross-data: XA1 transfer matrix (2x2), XA7, XA5.
- Phase 11: Q6 AA7 and AA8; Q7 AA13 (P in {128, 512, 2048}, N in {8, 16, 32}, one factor at a time); Q8 XA2 and XA6; Q9 FA16 sensitivity (detector r, baseline window, sustain steps, kappa, bandwidth tier, SLO threshold); Q10 X5 Chronos accuracy only; D4 and XA3 conditional.
- Interaction runs (G10) only if the pre-registered interaction_trigger fires after Q1; they are labelled exploratory.
- Compute fallback order, only when the dry-run estimate exceeds compute_budget_core_hours, each step by dated amendment before launch and reported: (1) X5 Chronos, (2) FA16 sensitivity, (3) XA6, (4) AA13 at P = 2048, (5) Q4, (6) Q3, (7) headroom 2.0 in the Must grid, (8) m = 2, (9) P = 256. If still over budget, the review queue Q11 is cut in the order RV3, RV2, RV1, RV0. Never cut: tests, the leak test, pre-registration, the clean-room rerun, replicates below min_replicates, unfavourable results.

## 9. Review-derived extensions (queue Q11, pre-registered here)

Interpretations marked (I) are the author's reading of the review text, fixed here.

- RV0: all seven policies (P0, P1, P2, P3, P4, P4a, P5) on D1, D2 and D3 separately at default m0 and h, 30 replicates per cell; P4a is also run on the Must-grid cells at default m0 and h so the seven-policy comparison is complete. P3 here uses the forecaster chosen for P4.
- RV1: contention ablation under cross-layer (Phase 9 must have passed), default m0 and h, moderate tier, phi 1.0, coupled mode, for P1, P3, P4a, P5. Rows: full model; CPU contention off; NIC contention off; both off, all else identical. (I) CPU contention off means kappa = 0 (no CPU overhead on moving partitions). NIC contention off means migration flows are not charged to links and the move duration is the constant m0. Both off removes both effects. Metrics: damage, violation rate, peak backlog, migration duration, recovery ratio.
- RV2: robustness to initial placement: balanced (hash) vs skewed, for P0, P1 and P4a. The definition of the skewed placement is deferred (section 10) and is fixed before Q11 is launched.
- RV3: migration throttling: (I) a throttled policy is the same policy at the lowest AA5 budget level (throttle_budget_B = 1) against the default budget (4), for P1 and P4a, on balanced and skewed placement.
- RV4: logging in every P4 and P4a row: point-forecast error, fixed-quantile and adaptive interval coverage, interval width, adaptive alpha and migration-trigger decisions.
- RV5: the regime grid (onset slope x migration duration) is Q2; forecast accuracy vs benefit uses the five forecasters of C4; sensitivity to migration and capacity parameters is Q1, Q4 and Q7. No extra runs.
- The review's pairwise comparison table (P4a against P1, P0, P3, P4, P5) is C13. The review's Table 8 has a cut-off sentence in the source; its definition is asked for in Phase 14.

## 10. Deferred values

Every null key in configs/default.yaml is fixed by a dated amendment, never from evaluation results, and always before any policy run on evaluation data. Each amendment states whether it precedes any tuning-split policy run.

| Keys | Fixed in | Rule or source |
| --- | --- | --- |
| detector.eps, detector.onset_class_cut, injector.* | Phase 3 | load scale of the training prefix; dataset median rise time |
| system.capacity_jitter_frac, system.wasted_move_window, tuning.plateau_tol, tuning.plateau_window | Phase 5, before the first tuning run | design values written before tuning |
| forecasters.lookback_steps, rolling_windows, ridge_alpha_grid, lightgbm_grid | Phase 6, before forecaster tuning | design values, tuned by MASE on the tuning split |
| tuning.hysteresis_levels, compute.runner_stall_minutes | Phase 7 | design value; user-declared |
| conformal.*, tuning.psi_grid, layers.d1_demand_field, d2_demand_c0, d2_demand_c1, demand_fallback, d1_bytes_sigma | Phase 9 | A15 read first; assumed values marked assumed |
| stats.interaction_trigger, review_extensions.skewed_placement | Phase 10, before Q1 or Q11 is launched | effect-size and interval rule; skew definition |
| system.step_rescaling_rule | Phase 11, before Q8 | rescaling of step-denominated parameters |

## 11. Freeze rule and clean-room

Results are frozen in Phase 13 (manifest, ENV.md, lock). No run after the freeze counts as pre-registered. The clean-room rerun rebuilds from raw files, retunes, regenerates forecasts and reruns the cells selected by cleanroom_stride (1 = every cell; every tenth cell in sorted cell-id order only if the estimate exceeds compute_budget_core_hours, by amendment). Rows and tuned.yaml must match byte for byte. Timing is logged separately from result rows.

## 12. Reporting rules

Unfavourable cells, cells under min_replicates and negative results are reported as findings. Work outside this file is labelled exploratory. The words first, novel and state of the art are not used unless supported; the survey statement is "not found in the surveyed literature". Every cross-layer statement says "under the stated model". Every assumed quantity is marked assumed in TA2. Nothing is tuned after looking at evaluation results.
