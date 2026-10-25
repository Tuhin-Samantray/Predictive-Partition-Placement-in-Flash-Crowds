# DATA_LOG.md

Retrieved 2026-10-23 by reading the primary pages listed below. No value here is from memory.
Exact sizes and hashes of downloaded files are appended in step 1.6.

## D1: Azure Functions Trace 2019
- Page: https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md
- Page revision stated: revision 2, 20200618. Data collected July 2019. Sampled per application; sampling rate unspecified.
- Download URL on the page (used): https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz
- Note: a search-result snippet of an older copy listed a different blob-storage URL. The current page URL is used.
- Archive format: .tar.xz
- Files per page: invocations_per_function_md.anon.d[01-14].csv; function_durations_percentiles.anon.d[01-14].csv; app_memory_percentiles.anon.d[01..12].csv (page says the last 2 of the 12 are missing). Used here: days 1 to 7 of all three families.
- Invocation schema: HashOwner, HashApp, HashFunction, Trigger, then 1440 columns named 1..1440 (invocations per minute of the 24 h file). Expected 1444 columns. Counts are recorded after the functions execute. Trigger groups: http, timer, event, queue, storage, orchestration, others.
- Duration schema: HashOwner, HashApp, HashFunction, Average, Count, Minimum, Maximum, percentile_Average_{0,1,25,50,75,99,100}. Milliseconds, excluding cold start. Percentiles are of 30-second averages, not of single invocations. Count can differ from the invocation files.
- Memory schema: HashOwner, HashApp, SampleCount, AverageAllocatedMb, AverageAllocatedMb_pct{1,5,25,50,75,95,99,100}. Application level only: there is no function column.
- IDs are hashed with HMAC-SHA256 and secret salts, consistent across file types.
- Licence: the page states CC-BY Attribution. LICENSE file text: see step 1.5.
- Citation (dataset page): Shahrad, Fonseca, Goiri, Chaudhry, Batum, Cooke, Laureano, Tresness, Russinovich, Bianchini. Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider. USENIX ATC 2020, Boston, July 2020.
- Open item: the execution guide does not define the role of the application-memory file. To be asked before Phase 3.

## D2: Twitter cache traces
- Repository: https://github.com/twitter/cache-trace (README read 2026-10-23)
- Content: Twemcache/Pelikan clusters, 54 clusters, collected Mar 2020, one week each.
- Format: zstd-compressed comma-separated text, one request per row, 7 fields: timestamp (seconds), anonymized key, key size (bytes), value size (bytes), client id, operation, TTL.
- Hosts listed: CMU PDL, SNIA, Storj, Baidu pan. Used here: https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source/
- Naming: CMU files are clusterN.sort.zst. The README describes a clusterN.0.zst split scheme and does not define "sort". Phase 3 checks that timestamps are non-decreasing instead of assuming it.
- Directory also holds sample10/, sample100/ and a sha1sum file. The samples are not used.
- Licence: CC BY 4.0. Full LICENSE text read in the repository on 2026-10-23.
- Citation (README): Yang, Yue, Rashmi Vinayak. A large scale analysis of hundreds of in-memory cache clusters at Twitter. OSDI 2020. Note: the guide's A11 writes "Rashmi"; use the README's form in Phase 14.
- Selection rule (fixed in PREREG): among clusters under d2_max_gb, take the largest file. d2_max_gb = 2 GiB (2147483648 bytes), unit chosen to match the server listing, set before any policy run.
- Compressed sizes in the CMU listing (rounded): cluster10 1.0G, cluster26 1.3G, cluster50 1.4G, cluster45 1.7G, cluster53 1.9G. Every other cluster is 4.5G or more.
- Attempt log (detector counts only, never policy results): attempt 1 = cluster53 (largest under the cap, pending exact byte check in step 1.1).
- D1 licence confirmed 2026-10-23: the LICENSE file in the Azure/AzurePublicDataset repository opens with "Attribution 4.0 International" (Creative Commons).
- D1 verification 2026-10-23: days 1 to 7 present for all three families (21 files); invocation header 1444 columns; duration and memory headers match the schemas above. The archive also contains days beyond 7; only days 1 to 7 are used.

## Downloaded files (appended 2026-10-23)
- D1 archive: azurefunctions-dataset2019.tar.xz, 142968140 bytes, SHA-256 aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89
- D1 CSVs: 40 files, hashes in logs\azure_sha256.txt
- D2 file: cluster53.sort.zst, 2084971699 bytes, SHA-256 d433e1b50c51594f8348966965fd0ebbeb0f90b884b5ddf1404ad919d232b3d9
- D2 SHA-256 list: logs\twitter_sha256.txt; CMU sha1sum listing saved at logs\twitter_sha1sum_listing.txt
- D2 integrity 2026-10-23: SHA-1 of cluster53.sort.zst matches the CMU sha1sum listing (041d6091d99bb228f60249a6d871dcfcd506d841). The listing also contains a different cluster53.zst entry, not used.
- D2 first rows 2026-10-23: 7 comma-separated fields per row as in the README; first five rows have timestamp 0; some get rows show value size 0. Ordering of timestamps and the meaning of value size 0 are to be checked in Phase 3.

## Phase 3 findings (2026-10-23, from logs\D1_load_stats.json and logs\D2_load_stats.json)
- D1: 307026 invocation-file rows over days 1 to 7; L is 10080 x 512; total invocations 6257408066; every partition has load and duration data. Application-memory files: schema checked only, not used in any result (option a, amendment A3).
- D2 cluster53.sort.zst: 246508262 rows; first timestamp 0; timestamps non-decreasing over the whole file; 18508 steps of 10 s (about 51.4 h) with no empty step. The README describes one week per cluster; the file in hand spans about 2.1 days. Cause not determined.
- D2 operations: get 217410986, prepend 22374671, set 6713419, delete 9186. Rows with value size 0: get 50168845, set 4440241, delete 9186. The README does not define value size 0; Bytes = key size + value size as the guide states.
- D2 selection rule: attempt 1 cluster53 gave 177 detector events, 102 on the evaluation span (min_events 24), so the rule keeps cluster53.
- Onset-class cut points (median rise time): D1 1.0 steps, D2 1.0 steps; recorded in AMENDMENTS.md.

## A15 (Gibbs and Candes), read 2026-10-24 12:10
- Source read: arXiv 2106.00170 (doi 10.48550/arXiv.2106.00170); proceedings entry Advances in Neural Information Processing Systems 34 (NeurIPS 2021), oral. Status M changed to V for authors, venue and year. The NeurIPS DOI was not seen and is not recorded.
- Update rule (their eq. 2): alpha_next = alpha + gamma x (alpha_target - err), where alpha is the miscoverage level (0.1 for 90 percent coverage) and err is 1 when the realized value falls outside the interval used at that step. Initial alpha equals the target in their experiments.
- Step size: their experiments use gamma = 0.005, chosen for stable alpha trajectories that still adapt; larger gamma adapts faster and is more volatile (their Sec 2.1).
- No clipping in the paper: alpha stays within [-gamma, 1 + gamma] without it (their Lemma 4.1). a_min and a_max are therefore this project's own bounds.
- Their feedback is immediate; delayed feedback is listed as an open problem in their discussion. Their long-run coverage guarantee (Proposition 4.1) is not claimed for P4a, whose feedback arrives m + d steps late.
- Their intervals use conformity scores and the empirical score quantile. P4a keeps P4's one-sided positive-residual quantile and only replaces its fixed level by the online level 1 - alpha.


## Chronos Bolt (Q10, X5), 2026-10-24
Source: the description shipped inside the installed chronos-forecasting 2.3.2 package (importlib metadata, saved to logs/chronos_readme.txt). Environment .venv_chronos: Python 3.14.3, torch 2.14.1 (CPU), transformers 5.20.2. Model id amazon/chronos-bolt-small, 48M parameters per the README table. Config read from the loaded model: context_length 2048, input_patch_size 16, prediction_length 64, quantiles 0.1 to 0.9 in steps of 0.1, pipeline class ChronosBoltPipeline; predict takes a (series, time) tensor and returns (batch, 9 quantiles, steps). The README describes Bolt as direct multi-step quantile forecasting and also lists Chronos-2 (amazon/chronos-2, 120M parameters, released 2025-10-20); Chronos-2 is not used (A50, A51). The licence of the model weights was not read from a primary source: check the model card before publication.
