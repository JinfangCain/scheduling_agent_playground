from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
SCHEDULE_DIR = OUTPUT_DIR / "schedules"

RULES = ["FCFS", "SPT", "EDD", "PRIORITY"]


def sort_jobs(jobs: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "FCFS":
        return jobs.sort_values(["arrival_time", "job_id"])
    if rule == "SPT":
        return jobs.sort_values(["processing_time", "arrival_time", "job_id"])
    if rule == "EDD":
        return jobs.sort_values(["due_date", "arrival_time", "job_id"])
    if rule == "PRIORITY":
        return jobs.sort_values(["priority", "due_date", "arrival_time", "job_id"], ascending=[False, True, True, True])
    raise ValueError(f"Unknown rule: {rule}")


def schedule_jobs(jobs: pd.DataFrame, machines: pd.DataFrame, rule: str) -> pd.DataFrame:
    machine_available = {
        row.machine_id: int(row.available_from)
        for row in machines.itertuples(index=False)
    }

    rows = []
    ordered_jobs = sort_jobs(jobs, rule)

    for job in ordered_jobs.itertuples(index=False):
        machine_id = min(machine_available, key=lambda mid: (machine_available[mid], mid))
        start = max(int(job.arrival_time), machine_available[machine_id])
        finish = start + int(job.processing_time)
        lateness = max(0, finish - int(job.due_date))
        flow_time = finish - int(job.arrival_time)

        rows.append(
            {
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
        )
        machine_available[machine_id] = finish

    return pd.DataFrame(rows).sort_values(["start_time", "machine_id", "job_id"])


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

