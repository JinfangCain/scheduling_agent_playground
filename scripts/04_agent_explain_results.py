from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"


def build_prompt(summary: pd.DataFrame, report_text: str) -> str:
    return f"""
You are a local scheduling research assistant.

Explain the synthetic scheduling experiment below. Do not claim this is a real semiconductor fab.
Focus on:
1. which dispatching rule performed best,
2. the trade-off among makespan, lateness, flow time, and utilization,
3. what a researcher should test next,
4. what limitations remain before applying this to semiconductor manufacturing.

Return concise markdown.

Summary table:
{summary.to_csv(index=False)}

Existing report:
{report_text}
""".strip()


def main() -> None:
    comparison_path = OUTPUT_DIR / "schedule_comparison.xlsx"
    report_path = OUTPUT_DIR / "comparison_report.md"

    if not comparison_path.exists() or not report_path.exists():
        raise FileNotFoundError("Run scripts/03_compare_results.py first.")

    summary = pd.read_excel(comparison_path, sheet_name="Summary")
    report_text = report_path.read_text(encoding="utf-8")
    prompt = build_prompt(summary, report_text)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        text = response.json().get("response", "").strip()
    except Exception as exc:
        text = (
            "# Local LLM Explanation Not Created\n\n"
            f"Ollama was not available or the request failed: `{exc}`.\n\n"
            "The rule-based scheduling outputs are still valid. Start Ollama and rerun this script if you want the optional explanation layer."
        )

    out_path = OUTPUT_DIR / "agent_explanation.md"
    out_path.write_text(text + "\n", encoding="utf-8")

    audit_path = OUTPUT_DIR / "agent_explanation_request.json"
    audit_path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "ollama_url": OLLAMA_URL,
                "input_files": [str(comparison_path), str(report_path)],
                "output_file": str(out_path),
                "note": "Synthetic data only. Local Ollama endpoint only.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {out_path}")
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()

