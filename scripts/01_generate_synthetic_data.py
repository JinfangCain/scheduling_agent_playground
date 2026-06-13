from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def build_jobs(n_jobs: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    for i in range(1, n_jobs + 1):
        arrival = rng.randint(0, 40)
        processing = rng.randint(3, 18)
        slack = rng.randint(8, 45)
        due = arrival + processing + slack
        priority = rng.choice([1, 1, 2, 2, 3])

        rows.append(
            {
                "job_id": f"J{i:03d}",
                "arrival_time": arrival,
                "processing_time": processing,
                "due_date": due,
                "priority": priority,
            }
        )

    return pd.DataFrame(rows).sort_values(["arrival_time", "job_id"])


def build_machines(n_machines: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine_id": [f"M{i:02d}" for i in range(1, n_machines + 1)],
            "available_from": [0 for _ in range(n_machines)],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic scheduling data.")
    parser.add_argument("--jobs", type=int, default=40, help="Number of synthetic jobs.")
    parser.add_argument("--machines", type=int, default=4, help="Number of parallel machines.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.jobs, args.seed)
    machines = build_machines(args.machines)

    jobs.to_csv(DATA_DIR / "synthetic_jobs.csv", index=False)
    machines.to_csv(DATA_DIR / "synthetic_machines.csv", index=False)

    print(f"Wrote {DATA_DIR / 'synthetic_jobs.csv'}")
    print(f"Wrote {DATA_DIR / 'synthetic_machines.csv'}")


if __name__ == "__main__":
    main()

