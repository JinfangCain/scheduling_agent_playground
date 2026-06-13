from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
SCHEDULE_DIR = OUTPUT_DIR / "schedules"
CHART_DIR = OUTPUT_DIR / "charts"


def summarize_schedule(schedule: pd.DataFrame) -> dict:
    rule = schedule["rule"].iloc[0]
    makespan = int(schedule["finish_time"].max())
    total_processing = int(schedule["processing_time"].sum())
    machine_count = schedule["machine_id"].nunique()
    utilization = total_processing / (makespan * machine_count) if makespan else 0

    return {
        "rule": rule,
        "jobs": len(schedule),
        "machines_used": machine_count,
        "makespan": makespan,
        "average_completion_time": round(float(schedule["finish_time"].mean()), 2),
        "average_flow_time": round(float(schedule["flow_time"].mean()), 2),
        "total_lateness": int(schedule["lateness"].sum()),
        "average_lateness": round(float(schedule["lateness"].mean()), 2),
        "late_jobs": int(schedule["is_late"].sum()),
        "machine_utilization": round(utilization, 3),
    }


def write_svg_gantt_chart(schedule: pd.DataFrame, out_path: Path) -> None:
    machines = sorted(schedule["machine_id"].unique())
    y_positions = {machine: i for i, machine in enumerate(machines)}
    makespan = int(schedule["finish_time"].max())
    width = 1100
    row_height = 36
    margin_left = 80
    margin_top = 42
    plot_width = width - margin_left - 30
    height = margin_top + row_height * len(machines) + 35

    def x_at(t: int) -> float:
        return margin_left + (t / max(makespan, 1)) * plot_width

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="24" font-family="Arial" font-size="18" font-weight="bold">Gantt Chart: {schedule["rule"].iloc[0]}</text>',
    ]

    for machine, y_index in y_positions.items():
        y = margin_top + y_index * row_height
        lines.append(f'<text x="15" y="{y + 22}" font-family="Arial" font-size="13">{machine}</text>')
        lines.append(f'<line x1="{margin_left}" y1="{y + 26}" x2="{width - 30}" y2="{y + 26}" stroke="#dddddd"/>')

    for row in schedule.itertuples(index=False):
        y = margin_top + y_positions[row.machine_id] * row_height + 5
        x = x_at(int(row.start_time))
        bar_width = max(3, x_at(int(row.finish_time)) - x)
        color = "#d95f02" if row.is_late else "#1b9e77"
        lines.append(
            f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="22" '
            f'fill="{color}" stroke="#222222" stroke-width="0.6"/>'
        )
        if bar_width >= 24:
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y + 15}" text-anchor="middle" '
                f'font-family="Arial" font-size="9" fill="white">{row.job_id}</text>'
            )

    lines.append(f'<text x="{margin_left}" y="{height - 10}" font-family="Arial" font-size="11">Green: on time or early. Orange: late.</text>')
    lines.append("</svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_gantt_chart(schedule: pd.DataFrame, out_path: Path) -> str:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        svg_path = out_path.with_suffix(".svg")
        write_svg_gantt_chart(schedule, svg_path)
        return str(svg_path)

    machines = sorted(schedule["machine_id"].unique())
    y_positions = {machine: i for i, machine in enumerate(machines)}

    fig, ax = plt.subplots(figsize=(12, 5))
    for row in schedule.itertuples(index=False):
        y = y_positions[row.machine_id]
        color = "#d95f02" if row.is_late else "#1b9e77"
        ax.barh(y, row.processing_time, left=row.start_time, color=color, edgecolor="black", alpha=0.85)
        ax.text(row.start_time + row.processing_time / 2, y, row.job_id, ha="center", va="center", fontsize=7)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(machines)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(f"Gantt Chart: {schedule['rule'].iloc[0]}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def write_report(summary: pd.DataFrame, best_rule: str, chart_path: str) -> None:
    lines = [
        "# Scheduling Comparison Report",
        "",
        "Synthetic single-stage parallel-machine scheduling experiment.",
        "",
        "Lower values are better for makespan, average flow time, total lateness, average lateness, and late jobs. Higher machine utilization is usually better, but only if lateness is not made worse.",
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
            "## Interpretation",
            "",
            "- `FCFS` is simple and fair by arrival order, but may perform poorly when long jobs block short urgent jobs.",
            "- `SPT` often reduces average flow time, but can delay urgent or high-priority jobs.",
            "- `EDD` directly targets due-date performance.",
            "- `PRIORITY` protects high-priority jobs, but can increase delay for low-priority jobs.",
            "",
            f"Gantt chart: `{chart_path}`.",
        ]
    )

    (OUTPUT_DIR / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    schedule_paths = sorted(SCHEDULE_DIR.glob("schedule_*.csv"))
    if not schedule_paths:
        raise FileNotFoundError("Run scripts/02_run_dispatch_rules.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    schedules = [pd.read_csv(path) for path in schedule_paths]
    summary = pd.DataFrame([summarize_schedule(schedule) for schedule in schedules])
    summary = summary.sort_values(["total_lateness", "makespan", "average_flow_time", "rule"])
    best_rule = summary.iloc[0]["rule"]

    comparison_path = OUTPUT_DIR / "schedule_comparison.xlsx"
    with pd.ExcelWriter(comparison_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        for schedule in schedules:
            rule = schedule["rule"].iloc[0]
            schedule.to_excel(writer, sheet_name=rule, index=False)

    best_schedule = next(schedule for schedule in schedules if schedule["rule"].iloc[0] == best_rule)
    chart_path = write_gantt_chart(best_schedule, CHART_DIR / f"gantt_{best_rule.lower()}.png")
    write_report(summary, best_rule, chart_path)

    print(f"Wrote {comparison_path}")
    print(f"Wrote {OUTPUT_DIR / 'comparison_report.md'}")
    print(f"Wrote {chart_path}")


if __name__ == "__main__":
    main()
