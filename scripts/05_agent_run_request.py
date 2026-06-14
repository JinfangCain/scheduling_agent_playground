from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
AGENT_RUNS_DIR = OUTPUT_DIR / "agent_runs"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = "gpt-4.1-mini"
KNOWN_RULES = ["FCFS", "SPT", "EDD", "PRIORITY"]


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dispatch = load_module(BASE_DIR / "scripts" / "02_run_dispatch_rules.py", "dispatch_rules")
compare = load_module(BASE_DIR / "scripts" / "03_compare_results.py", "compare_results")


def build_extraction_prompt(request_text: str) -> str:
    return f"""
Extract a scheduling request from the text below.

Return JSON only with this schema:
{{
  "machines": [
    {{"machine_id": "M01", "available_from": 0}}
  ],
  "jobs": [
    {{
      "job_id": "J1",
      "arrival_time": 0,
      "processing_time": 5,
      "due_date": 14,
      "priority": 2
    }}
  ],
  "rules": ["FCFS", "SPT", "EDD", "PRIORITY"],
  "objective": "minimize total lateness"
}}

Use only values explicitly present in the request. If the request gives a
machine count and says all machines are available at time 0, create machine ids
M01, M02, and so on. Do not invent jobs.

Request:
{request_text}
""".strip()


def ask_ollama_for_json(request_text: str, model: str, ollama_url: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": build_extraction_prompt(request_text),
        "stream": False,
        "format": "json",
    }
    response = requests.post(ollama_url, json=payload, timeout=120)
    response.raise_for_status()
    raw_text = response.json().get("response", "").strip()
    return json.loads(raw_text)


def extract_response_text(data: dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"])

    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if "text" in content:
                chunks.append(str(content["text"]))
            elif "output_text" in content:
                chunks.append(str(content["output_text"]))
    return "\n".join(chunks).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def ask_openai_for_json(
    request_text: str,
    model: str,
    api_key: str,
    responses_url: str = OPENAI_RESPONSES_URL,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    response = requests.post(
        responses_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": build_extraction_prompt(request_text),
        },
        timeout=120,
    )
    response.raise_for_status()
    raw_text = extract_response_text(response.json())
    return parse_json_object(raw_text)


def fallback_parse_request(request_text: str) -> dict[str, Any]:
    machine_match = re.search(r"(\d+)\s+machines?\b", request_text, re.IGNORECASE)
    machine_count = int(machine_match.group(1)) if machine_match else 1

    jobs = []
    job_pattern = re.compile(
        r"Job\s+([A-Za-z0-9_-]+)\s+arrives?\s+at\s+(\d+)"
        r".*?processing\s+time\s+(\d+)"
        r".*?due\s+date\s+(\d+)"
        r".*?priority\s+(\d+)",
        re.IGNORECASE,
    )
    for match in job_pattern.finditer(request_text):
        jobs.append(
            {
                "job_id": match.group(1),
                "arrival_time": int(match.group(2)),
                "processing_time": int(match.group(3)),
                "due_date": int(match.group(4)),
                "priority": int(match.group(5)),
            }
        )

    upper_text = request_text.upper()
    rules = [rule for rule in KNOWN_RULES if rule in upper_text]
    if not rules:
        rules = KNOWN_RULES.copy()

    return {
        "machines": [
            {"machine_id": f"M{i:02d}", "available_from": 0}
            for i in range(1, machine_count + 1)
        ],
        "jobs": jobs,
        "rules": rules,
        "objective": "minimize total lateness",
    }


def normalize_and_validate(parsed: dict[str, Any]) -> dict[str, Any]:
    if "machines" not in parsed and "machine_count" in parsed:
        parsed["machines"] = [
            {"machine_id": f"M{i:02d}", "available_from": 0}
            for i in range(1, int(parsed["machine_count"]) + 1)
        ]

    jobs = parsed.get("jobs")
    machines = parsed.get("machines")
    rules = parsed.get("rules") or KNOWN_RULES

    if not isinstance(jobs, list) or not jobs:
        raise ValueError("No jobs were found in the request.")
    if not isinstance(machines, list) or not machines:
        raise ValueError("No machines were found in the request.")

    clean_jobs = []
    seen_jobs = set()
    for job in jobs:
        missing = {"job_id", "arrival_time", "processing_time", "due_date", "priority"} - set(job)
        if missing:
            raise ValueError(f"Job is missing required fields: {sorted(missing)}")
        job_id = str(job["job_id"])
        if job_id in seen_jobs:
            raise ValueError(f"Duplicate job_id: {job_id}")
        seen_jobs.add(job_id)

        arrival_time = int(job["arrival_time"])
        processing_time = int(job["processing_time"])
        due_date = int(job["due_date"])
        priority = int(job["priority"])
        if arrival_time < 0:
            raise ValueError(f"{job_id} has negative arrival_time.")
        if processing_time <= 0:
            raise ValueError(f"{job_id} has non-positive processing_time.")

        clean_jobs.append(
            {
                "job_id": job_id,
                "arrival_time": arrival_time,
                "processing_time": processing_time,
                "due_date": due_date,
                "priority": priority,
            }
        )

    clean_machines = []
    seen_machines = set()
    for machine in machines:
        missing = {"machine_id", "available_from"} - set(machine)
        if missing:
            raise ValueError(f"Machine is missing required fields: {sorted(missing)}")
        machine_id = str(machine["machine_id"])
        if machine_id in seen_machines:
            raise ValueError(f"Duplicate machine_id: {machine_id}")
        seen_machines.add(machine_id)
        available_from = int(machine["available_from"])
        if available_from < 0:
            raise ValueError(f"{machine_id} has negative available_from.")
        clean_machines.append({"machine_id": machine_id, "available_from": available_from})

    clean_rules = []
    for rule in rules:
        rule_name = str(rule).upper()
        if rule_name not in KNOWN_RULES:
            raise ValueError(f"Unknown scheduling rule: {rule}")
        if rule_name not in clean_rules:
            clean_rules.append(rule_name)

    return {
        "jobs": clean_jobs,
        "machines": clean_machines,
        "rules": clean_rules,
        "objective": str(parsed.get("objective", "minimize total lateness")),
    }


def next_run_dir(request_path: Path | str) -> Path:
    AGENT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stem = request_path.stem if isinstance(request_path, Path) else str(request_path)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "request"
    for i in range(1, 1000):
        candidate = AGENT_RUNS_DIR / f"{stem}_{i:03d}"
        if not candidate.exists():
            return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return AGENT_RUNS_DIR / f"{stem}_{timestamp}"


def write_agent_report(summary: pd.DataFrame, best_rule: str, chart_path: str, out_path: Path) -> None:
    try:
        chart_display_path = str(Path(chart_path).resolve().relative_to(BASE_DIR))
    except ValueError:
        chart_display_path = str(chart_path)

    lines = [
        "# Agent Scheduling Comparison Report",
        "",
        "Natural-language scheduling request converted into a transparent rule-based scheduling experiment.",
        "",
        f"Best rule by total lateness, then makespan: **{best_rule}**.",
        "",
        "## Summary Table",
        "",
        "| Rule | Makespan | Avg Flow | Total Lateness | Late Jobs | Utilization |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.rule} | {row.makespan} | {row.average_flow_time} | "
            f"{row.total_lateness} | {row.late_jobs} | {row.machine_utilization} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The LLM, when available, is used only to parse and explain the request.",
            "- Scheduling decisions are made by explicit dispatching rules.",
            f"- Gantt chart: `{chart_display_path}`.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def deterministic_summary(summary: pd.DataFrame, objective: str) -> str:
    best = summary.iloc[0]
    lines = [
        "# Natural Language Summary",
        "",
        f"I compared {', '.join(summary['rule'].tolist())} for this request.",
        "",
        f"The recommended rule is **{best.rule}** for the objective: {objective}.",
        "",
        f"{best.rule} had total lateness {best.total_lateness}, makespan {best.makespan}, "
        f"average flow time {best.average_flow_time}, and {best.late_jobs} late jobs.",
        "",
        "Trade-off notes:",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.rule}: makespan {row.makespan}, average flow {row.average_flow_time}, "
            f"total lateness {row.total_lateness}, late jobs {row.late_jobs}, "
            f"utilization {row.machine_utilization}."
        )

    lines.extend(
        [
            "",
            "Visualization: the app displays a Gantt chart for the recommended rule below the comparison table. Use it to inspect machine assignment, job order, idle time, and whether any jobs are late.",
            "",
            "This is a transparent rule-based recommendation. It should be treated as a baseline experiment, not a final production scheduler.",
        ]
    )
    return "\n".join(lines) + "\n"


def ask_ollama_for_summary(summary: pd.DataFrame, report_text: str, model: str, ollama_url: str) -> str:
    prompt = f"""
You are a local scheduling research assistant.

Explain this scheduling run in concise markdown for a collaborator demo.
Focus on the recommended rule, trade-offs, and limitations.
Do not claim this is a real factory scheduler.
The app displays a Gantt chart for the recommended rule below the comparison table.
Do not say that Gantt charts or visualizations are not included.

Summary:
{summary.to_csv(index=False)}

Report:
{report_text}
""".strip()
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(ollama_url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip() + "\n"


def build_summary_prompt(summary: pd.DataFrame, report_text: str) -> str:
    return f"""
You are a scheduling research assistant.

Explain this scheduling run in concise markdown for a collaborator demo.
Focus on the recommended rule, trade-offs, and limitations.
Do not claim this is a real factory scheduler.
The app displays a Gantt chart for the recommended rule below the comparison table.
Do not say that Gantt charts or visualizations are not included.
If you mention the Gantt chart, explain that it shows machine assignment over time,
job order, idle gaps, and late jobs if any appear.

Summary:
{summary.to_csv(index=False)}

Report:
{report_text}
""".strip()


def ask_openai_for_summary(
    summary: pd.DataFrame,
    report_text: str,
    model: str,
    api_key: str,
    responses_url: str = OPENAI_RESPONSES_URL,
) -> str:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    response = requests.post(
        responses_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": build_summary_prompt(summary, report_text),
        },
        timeout=120,
    )
    response.raise_for_status()
    return extract_response_text(response.json()).strip() + "\n"


def run_agent_text(
    request_text: str,
    run_stem: str,
    use_ollama: bool = False,
    model: str = MODEL,
    ollama_url: str = OLLAMA_URL,
    provider: str = "deterministic",
    openai_api_key: str | None = None,
    openai_model: str = OPENAI_MODEL,
    openai_responses_url: str = OPENAI_RESPONSES_URL,
) -> dict[str, Any]:
    parser_used = "fallback_regex"
    parser_error = None
    provider = provider.lower()
    if use_ollama:
        provider = "ollama"

    if provider == "openai":
        try:
            parsed = ask_openai_for_json(
                request_text=request_text,
                model=openai_model,
                api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
                responses_url=openai_responses_url,
            )
            parser_used = f"openai:{openai_model}"
        except Exception as exc:
            parser_error = str(exc)
            parsed = fallback_parse_request(request_text)
    elif provider == "ollama":
        try:
            parsed = ask_ollama_for_json(request_text, model, ollama_url)
            parser_used = f"ollama:{model}"
        except Exception as exc:
            parser_error = str(exc)
            parsed = fallback_parse_request(request_text)
    elif provider == "deterministic":
        parsed = fallback_parse_request(request_text)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    request = normalize_and_validate(parsed)
    run_dir = next_run_dir(run_stem)
    schedule_dir = run_dir / "schedules"
    chart_dir = run_dir / "charts"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "request.txt").write_text(request_text, encoding="utf-8")
    (run_dir / "parsed_request.json").write_text(
        json.dumps(
            {
                "parser_used": parser_used,
                "parser_error": parser_error,
                "request": request,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    jobs = pd.DataFrame(request["jobs"]).sort_values(["arrival_time", "job_id"])
    machines = pd.DataFrame(request["machines"])
    jobs.to_csv(run_dir / "jobs.csv", index=False)
    machines.to_csv(run_dir / "machines.csv", index=False)

    schedules = []
    for rule in request["rules"]:
        schedule = dispatch.schedule_jobs(jobs, machines, rule)
        schedule.to_csv(schedule_dir / f"schedule_{rule.lower()}.csv", index=False)
        schedules.append(schedule)

    summary = pd.DataFrame([compare.summarize_schedule(schedule) for schedule in schedules])
    summary = summary.sort_values(["total_lateness", "makespan", "average_flow_time", "rule"])
    best_rule = summary.iloc[0]["rule"]

    comparison_path = run_dir / "schedule_comparison.xlsx"
    with pd.ExcelWriter(comparison_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        for schedule in schedules:
            rule = schedule["rule"].iloc[0]
            schedule.to_excel(writer, sheet_name=rule, index=False)

    best_schedule = next(schedule for schedule in schedules if schedule["rule"].iloc[0] == best_rule)
    chart_path = compare.write_gantt_chart(best_schedule, chart_dir / f"gantt_{best_rule.lower()}.png")
    report_path = run_dir / "comparison_report.md"
    write_agent_report(summary, best_rule, chart_path, report_path)

    report_text = report_path.read_text(encoding="utf-8")
    if provider == "openai":
        try:
            nl_summary = ask_openai_for_summary(
                summary=summary,
                report_text=report_text,
                model=openai_model,
                api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
                responses_url=openai_responses_url,
            )
        except Exception as exc:
            nl_summary = deterministic_summary(summary, request["objective"])
            nl_summary += f"\n_OpenAI summary unavailable: `{exc}`._\n"
    elif provider == "ollama":
        try:
            nl_summary = ask_ollama_for_summary(summary, report_text, model, ollama_url)
        except Exception as exc:
            nl_summary = deterministic_summary(summary, request["objective"])
            nl_summary += f"\n_Ollama summary unavailable: `{exc}`._\n"
    else:
        nl_summary = deterministic_summary(summary, request["objective"])
    (run_dir / "nl_summary.md").write_text(nl_summary, encoding="utf-8")

    print(f"Parsed {len(jobs)} jobs and {len(machines)} machines.")
    print(f"Parser used: {parser_used}")
    if parser_error:
        print(f"Parser fallback reason: {parser_error}")
    print(f"Compared {', '.join(request['rules'])}.")
    print(f"Best rule by total lateness, then makespan: {best_rule}.")
    print(f"Wrote {run_dir}")
    return {
        "run_dir": run_dir,
        "request": request,
        "jobs": jobs,
        "machines": machines,
        "summary": summary,
        "best_rule": best_rule,
        "chart_path": Path(chart_path),
        "report_path": report_path,
        "nl_summary_path": run_dir / "nl_summary.md",
        "comparison_path": comparison_path,
        "parser_used": parser_used,
        "parser_error": parser_error,
        "provider": provider,
    }


def run_agent_request(
    request_path: Path,
    use_ollama: bool,
    model: str,
    ollama_url: str,
    provider: str = "deterministic",
    openai_api_key: str | None = None,
    openai_model: str = OPENAI_MODEL,
) -> Path:
    result = run_agent_text(
        request_text=request_path.read_text(encoding="utf-8"),
        run_stem=request_path.stem,
        use_ollama=use_ollama,
        model=model,
        ollama_url=ollama_url,
        provider=provider,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
    )
    run_dir = result["run_dir"]
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an agent-style scheduling request from natural language.")
    parser.add_argument("request_path", type=Path, help="Text file containing the natural-language scheduling request.")
    parser.add_argument(
        "--provider",
        choices=["deterministic", "openai", "ollama"],
        default="deterministic",
        help="LLM provider for parsing and summary. Falls back to deterministic parsing on provider failure.",
    )
    parser.add_argument("--no-ollama", action="store_true", help="Deprecated alias for --provider deterministic.")
    parser.add_argument("--model", default=MODEL, help="Local Ollama model name.")
    parser.add_argument("--ollama-url", default=OLLAMA_URL, help="Local Ollama generate endpoint.")
    parser.add_argument("--openai-model", default=OPENAI_MODEL, help="OpenAI model name.")
    args = parser.parse_args()

    if not args.request_path.exists():
        raise FileNotFoundError(args.request_path)

    run_agent_request(
        request_path=args.request_path,
        use_ollama=False if args.no_ollama else args.provider == "ollama",
        model=args.model,
        ollama_url=args.ollama_url,
        provider="deterministic" if args.no_ollama else args.provider,
        openai_model=args.openai_model,
    )


if __name__ == "__main__":
    main()
