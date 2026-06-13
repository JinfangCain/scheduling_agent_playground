# Scheduling Comparison Report

Synthetic single-stage parallel-machine scheduling experiment.

Lower values are better for makespan, average flow time, total lateness, average lateness, and late jobs. Higher machine utilization is usually better, but only if lateness is not made worse.

Best rule by total lateness, then makespan: **EDD**.

## Summary Table

| Rule | Makespan | Avg Flow | Total Lateness | Late Jobs | Utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| EDD | 103 | 29.9 | 69 | 12 | 0.825 |
| FCFS | 96 | 27.35 | 122 | 12 | 0.885 |
| SPT | 116 | 36.52 | 433 | 16 | 0.733 |
| PRIORITY | 122 | 49.65 | 808 | 20 | 0.697 |

## Interpretation

- `FCFS` is simple and fair by arrival order, but may perform poorly when long jobs block short urgent jobs.
- `SPT` often reduces average flow time, but can delay urgent or high-priority jobs.
- `EDD` directly targets due-date performance.
- `PRIORITY` protects high-priority jobs, but can increase delay for low-priority jobs.

Gantt chart: `/Users/wang/Documents/Codex/Artificial-Intelligence/scheduling_agent_playground/outputs/charts/gantt_edd.svg`.