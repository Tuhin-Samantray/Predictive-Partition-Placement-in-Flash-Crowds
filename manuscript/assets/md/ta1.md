| dataset | role | step_seconds | n_steps | span_days | n_partitions | train_end | tune_end | eval_steps | n_events | events_eval_abrupt | events_eval_ramped | events_eval_all | bytes_provenance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | primary real trace | 60 | 10080 | 7 | 512 | 2880 | 4320 | 5760 | 6482 | 964 | 2743 | 0 | assumed |
| D2 | second real trace | 10 | 18508 | 2.142 | 512 | 4627 | 7403 | 11105 | 177 | 42 | 60 | 0 | real |
| D1_D3 | D1 with injected events | 60 | 10080 | 7 | 512 | 2880 | 4320 | 5760 | 40 | 0 | 0 | 40 | assumed |
| D2_D3 | D2 with injected events | 10 | 18508 | 2.142 | 512 | 4627 | 7403 | 11105 | 40 | 0 | 0 | 40 | real |
