# Scheduling Agent Playground

This is a local, synthetic-data playground for learning and researching scheduling agents. The first version is intentionally modest: it studies a single-stage parallel-machine scheduling problem with simple dispatching rules.

The goal is not to automate a real factory schedule. The goal is to build a transparent research prototype that can later be extended toward semiconductor manufacturing use cases.

## First Problem

Synthetic jobs arrive over time and must be assigned to one of several machines.

Each job has:

- a job id
- an arrival time
- a processing time
- a due date
- a priority

Each machine has:

- a machine id
- an availability time

The scripts compare simple scheduling rules:

- `FCFS`: first come, first served
- `SPT`: shortest processing time first
- `EDD`: earliest due date first
- `PRIORITY`: highest priority first, then earliest due date

## Outputs

The pipeline creates:

- synthetic job and machine CSV files
- one schedule CSV for each dispatching rule
- an Excel comparison workbook
- a markdown report
- a Gantt chart, if `matplotlib` is installed
- an optional local Ollama explanation report

## Run

From this folder:

```bash
python scripts/01_generate_synthetic_data.py
python scripts/02_run_dispatch_rules.py
python scripts/03_compare_results.py
```

Optional local LLM explanation:

```bash
python scripts/04_agent_explain_results.py
```

The Ollama script uses `http://localhost:11434/api/generate` and does not send data to an external API. It is optional; the scheduling playground works without any LLM.

## Design Principle

The scheduling algorithm remains transparent and rule-based. The LLM, when used, is only an explanation and research-assistant layer. It should help interpret results, suggest next experiments, and identify bottlenecks, but it should not silently decide the schedule.

