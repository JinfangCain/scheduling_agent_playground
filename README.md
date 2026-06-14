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

## Agent-Style Natural Language Demo

The first agent-style interface reads a natural-language request, extracts jobs and machines, validates the structured data, runs the same transparent dispatching rules, and writes a run-specific audit folder.

Local chatbot-style app:

```bash
streamlit run app.py
```

Deterministic demo without Ollama:

```bash
python scripts/05_agent_run_request.py examples/demo_request.txt --no-ollama
```

Local Ollama-assisted demo:

```bash
python scripts/05_agent_run_request.py examples/demo_request.txt
```

Each run writes outputs under `outputs/agent_runs/`, including the original request, parsed JSON, job and machine CSVs, schedule CSVs, an Excel comparison workbook, a markdown comparison report, a Gantt chart, and a natural-language summary.

## Streamlit Cloud Deployment

This repo is prepared for Streamlit Community Cloud:

- app entry point: `app.py`
- Python dependencies: `requirements.txt`
- Streamlit config: `.streamlit/config.toml`
- generated agent runs are ignored by Git: `outputs/agent_runs/`

For the first public demo, keep `Use local Ollama` off. Local Ollama points to `localhost`, which only works on the machine running Ollama. A future OpenAI-backed mode should read `OPENAI_API_KEY` from Streamlit secrets or the environment, never from committed source files.

In Streamlit Community Cloud, add secrets in the app settings using the shape shown in `.streamlit/secrets.example.toml`.

## Design Principle

The scheduling algorithm remains transparent and rule-based. The LLM, when used, is only an explanation and research-assistant layer. It should help interpret results, suggest next experiments, and identify bottlenecks, but it should not silently decide the schedule.
