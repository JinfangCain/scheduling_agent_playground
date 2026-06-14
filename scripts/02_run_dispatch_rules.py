from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
SCHEDULE_DIR = OUTPUT_DIR / "schedules"

RULES = ["FCFS", "SPT", "EDD", "PRIORITY", "OPTIMIZER"]


def sort_jobs(jobs: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "FCFS":
        return jobs.sort_values(["arrival_time", "job_id"])
    if rule == "SPT":
        return jobs.sort_values(["processing_time", "arrival_time", "job_id"])
    if rule == "EDD":
        return jobs.sort_values(["due_date", "arrival_time", "job_id"])
    if rule == "PRIORITY":
        return jobs.sort_values(["priority", "due_date", "arrival_time", "job_id"], ascending=[False, True, True, True])
    if rule == "OPTIMIZER":
        return jobs.sort_values(["due_date", "priority", "processing_time", "arrival_time", "job_id"], ascending=[True, False, True, True, True])
    raise ValueError(f"Unknown rule: {rule}")


def parse_jsonish(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]


def machine_ids(machines: pd.DataFrame) -> list[str]:
    return [str(row.machine_id) for row in machines.itertuples(index=False)]


def job_eligible_machines(job: Any, machines: pd.DataFrame) -> list[str]:
    all_machines = machine_ids(machines)
    if hasattr(job, "eligible_machines"):
        eligible = parse_jsonish(job.eligible_machines, all_machines)
        eligible = [str(machine_id) for machine_id in eligible]
        eligible = [machine_id for machine_id in eligible if machine_id in all_machines]
        return eligible or all_machines
    return all_machines


def machine_downtime_map(machines: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    downtime_by_machine: dict[str, list[tuple[int, int]]] = {}
    for row in machines.itertuples(index=False):
        windows = []
        if hasattr(row, "downtime_windows"):
            raw_windows = parse_jsonish(row.downtime_windows, [])
            for window in raw_windows:
                if isinstance(window, dict):
                    start = int(window.get("start", window.get("from", 0)))
                    end = int(window.get("end", window.get("to", start)))
                else:
                    start = int(window[0])
                    end = int(window[1])
                if end > start:
                    windows.append((start, end))
        downtime_by_machine[str(row.machine_id)] = sorted(windows)
    return downtime_by_machine


def earliest_feasible_start(ready_time: int, processing_time: int, downtime_windows: list[tuple[int, int]]) -> int:
    start = ready_time
    while True:
        shifted = False
        finish = start + processing_time
        for down_start, down_end in downtime_windows:
            if start < down_end and finish > down_start:
                start = down_end
                shifted = True
                break
        if not shifted:
            return start


def job_weight(job: Any) -> int:
    if hasattr(job, "weight"):
        try:
            return max(1, int(job.weight))
        except (TypeError, ValueError):
            return 1
    if hasattr(job, "priority"):
        return max(1, int(job.priority))
    return 1


def build_schedule_row(rule: str, job: Any, machine_id: str, start: int, finish: int) -> dict:
    lateness = max(0, finish - int(job.due_date))
    flow_time = finish - int(job.arrival_time)
    row = {
        "rule": rule,
        "job_id": job.job_id,
        "machine_id": machine_id,
        "arrival_time": int(job.arrival_time),
        "start_time": start,
        "finish_time": finish,
        "processing_time": int(job.processing_time),
        "due_date": int(job.due_date),
        "priority": int(job.priority),
        "flow_time": flow_time,
        "lateness": lateness,
        "is_late": lateness > 0,
    }
    if hasattr(job, "weight"):
        weight = job_weight(job)
        row["weight"] = weight
        row["weighted_lateness"] = lateness * weight
    if hasattr(job, "family"):
        row["family"] = job.family
    if hasattr(job, "eligible_machines"):
        row["eligible_machines"] = json.dumps(parse_jsonish(job.eligible_machines, []))
    return row


def schedule_jobs(jobs: pd.DataFrame, machines: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "OPTIMIZER":
        return schedule_jobs_optimizer(jobs, machines)

    machine_available = {
        row.machine_id: int(row.available_from)
        for row in machines.itertuples(index=False)
    }
    downtime_by_machine = machine_downtime_map(machines)

    rows = []
    ordered_jobs = sort_jobs(jobs, rule)

    for job in ordered_jobs.itertuples(index=False):
        candidates = []
        for machine_id in job_eligible_machines(job, machines):
            ready = max(int(job.arrival_time), machine_available[machine_id])
            start = earliest_feasible_start(ready, int(job.processing_time), downtime_by_machine.get(machine_id, []))
            finish = start + int(job.processing_time)
            candidates.append((finish, start, machine_id))
        if not candidates:
            raise ValueError(f"No eligible machines for job {job.job_id}.")

        finish, start, machine_id = min(candidates)
        rows.append(build_schedule_row(rule, job, machine_id, start, finish))
        machine_available[machine_id] = finish

    return pd.DataFrame(rows).sort_values(["start_time", "machine_id", "job_id"])


def schedule_jobs_optimizer(jobs: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    if len(jobs) <= 9:
        return schedule_jobs_search(jobs, machines)
    optimized = schedule_jobs(jobs, machines, "EDD")
    optimized["rule"] = "OPTIMIZER"
    return optimized


def schedule_jobs_search(jobs: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    job_records = list(jobs.sort_values(["arrival_time", "due_date", "job_id"]).itertuples(index=False))
    initial_available = {
        str(row.machine_id): int(row.available_from)
        for row in machines.itertuples(index=False)
    }
    downtime_by_machine = machine_downtime_map(machines)
    best: dict[str, Any] = {"score": None, "rows": None}

    def score_rows(rows: list[dict]) -> tuple[int, int, int, float]:
        total_weighted_lateness = sum(int(row.get("weighted_lateness", row["lateness"])) for row in rows)
        total_lateness = sum(int(row["lateness"]) for row in rows)
        late_jobs = sum(1 for row in rows if row["is_late"])
        makespan = max(int(row["finish_time"]) for row in rows) if rows else 0
        return (total_weighted_lateness, total_lateness, late_jobs, makespan)

    def lower_bound(rows: list[dict]) -> tuple[int, int, int, float]:
        if not rows:
            return (0, 0, 0, 0)
        total_weighted_lateness = sum(int(row.get("weighted_lateness", row["lateness"])) for row in rows)
        total_lateness = sum(int(row["lateness"]) for row in rows)
        late_jobs = sum(1 for row in rows if row["is_late"])
        makespan = max(int(row["finish_time"]) for row in rows)
        return (total_weighted_lateness, total_lateness, late_jobs, makespan)

    def search(remaining: list[Any], machine_available: dict[str, int], rows: list[dict]) -> None:
        if best["score"] is not None and lower_bound(rows) >= best["score"]:
            return
        if not remaining:
            score = score_rows(rows)
            if best["score"] is None or score < best["score"]:
                best["score"] = score
                best["rows"] = rows.copy()
            return

        ordered_remaining = sorted(
            remaining,
            key=lambda job: (int(job.due_date), -job_weight(job), int(job.processing_time), str(job.job_id)),
        )
        for job in ordered_remaining:
            rest = [candidate for candidate in remaining if candidate.job_id != job.job_id]
            candidates = []
            for machine_id in job_eligible_machines(job, machines):
                ready = max(int(job.arrival_time), machine_available[machine_id])
                start = earliest_feasible_start(ready, int(job.processing_time), downtime_by_machine.get(machine_id, []))
                finish = start + int(job.processing_time)
                row = build_schedule_row("OPTIMIZER", job, machine_id, start, finish)
                candidates.append((row.get("weighted_lateness", row["lateness"]), finish, start, machine_id, row))

            for _, finish, _, machine_id, row in sorted(candidates):
                next_available = machine_available.copy()
                next_available[machine_id] = finish
                search(rest, next_available, rows + [row])

    search(job_records, initial_available, [])
    if best["rows"] is None:
        raise ValueError("Optimizer could not build a feasible schedule.")
    return pd.DataFrame(best["rows"]).sort_values(["start_time", "machine_id", "job_id"])


def main() -> None:
    jobs_path = DATA_DIR / "synthetic_jobs.csv"
    machines_path = DATA_DIR / "synthetic_machines.csv"

    if not jobs_path.exists() or not machines_path.exists():
        raise FileNotFoundError("Run scripts/01_generate_synthetic_data.py first.")

    jobs = pd.read_csv(jobs_path)
    machines = pd.read_csv(machines_path)
    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)

    for rule in RULES:
        schedule = schedule_jobs(jobs, machines, rule)
        out_path = SCHEDULE_DIR / f"schedule_{rule.lower()}.csv"
        schedule.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
