COMPLETE

feat(labels): add next-day log-return labeling with dead-zone (ε=0.001) and per-ticker handling

- src/smc/labels.py: compute_next_log_return, label_from_returns, make_labels, save_labels
- tests/test_labels.py: unit tests for dead-zone and multi-ticker grouping
- update config.yaml example: label.horizon_days, label.epsilon





added setup.cfg
install pytest