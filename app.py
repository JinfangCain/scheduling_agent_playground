from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DEMO_REQUEST_PATH = BASE_DIR / "examples" / "demo_request.txt"
AGENT_SCRIPT_PATH = BASE_DIR / "scripts" / "05_agent_run_request.py"


def load_agent_module() -> Any:
    spec = importlib.util.spec_from_file_location("agent_run_request", AGENT_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent runner from {AGENT_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = load_agent_module()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def initialize_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Paste a scheduling request with jobs, machines, processing times, due dates, "
                    "and priorities. I will parse it, run transparent scheduling rules, and save the results."
                ),
            }
        ]
    if "request_text" not in st.session_state:
        st.session_state.request_text = read_text(DEMO_REQUEST_PATH)
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def run_request(request_text: str, use_ollama: bool, model: str, ollama_url: str) -> dict[str, Any]:
    return agent.run_agent_text(
        request_text=request_text,
        run_stem="streamlit_request",
        use_ollama=use_ollama,
        model=model,
        ollama_url=ollama_url,
    )


def assistant_error_message(exc: Exception) -> str:
    text = str(exc)
    if "No jobs were found" in text:
        return (
            "I could not find complete job records yet. Please include each job with arrival time, "
            "processing time, due date, and priority."
        )
    if "No machines were found" in text:
        return "I need the number of machines, or explicit machine IDs and availability times."
    if "missing required fields" in text:
        return f"I found an incomplete record: {text}. Please add the missing value and run again."
    return f"I could not run the scheduling request yet: {text}"


def render_result(result: dict[str, Any]) -> None:
    run_dir = Path(result["run_dir"])
    nl_summary_path = Path(result["nl_summary_path"])
    report_path = Path(result["report_path"])
    comparison_path = Path(result["comparison_path"])
    chart_path = Path(result["chart_path"])

    st.subheader("Recommendation")
    st.markdown(read_text(nl_summary_path))

    st.subheader("Comparison")
    summary = result["summary"]
    st.dataframe(summary, width="stretch", hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Rule", str(result["best_rule"]))
    col2.metric("Jobs", str(len(result["jobs"])))
    col3.metric("Machines", str(len(result["machines"])))

    st.subheader("Parsed Request")
    with st.expander("Jobs and machines", expanded=False):
        st.caption(f"Parser used: {result['parser_used']}")
        if result["parser_error"]:
            st.warning(f"Parser fallback reason: {result['parser_error']}")
        st.write("Jobs")
        st.dataframe(result["jobs"], width="stretch", hide_index=True)
        st.write("Machines")
        st.dataframe(result["machines"], width="stretch", hide_index=True)

    st.subheader("Gantt Chart")
    if chart_path.suffix.lower() == ".svg":
        st.image(str(chart_path))
    else:
        st.image(str(chart_path))

    st.subheader("Saved Outputs")
    st.code(display_path(run_dir))

    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download Excel",
        data=read_bytes(comparison_path),
        file_name="schedule_comparison.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    download_cols[1].download_button(
        "Download Report",
        data=read_text(report_path),
        file_name="comparison_report.md",
        mime="text/markdown",
    )
    download_cols[2].download_button(
        "Download Summary",
        data=read_text(nl_summary_path),
        file_name="nl_summary.md",
        mime="text/markdown",
    )


def main() -> None:
    st.set_page_config(page_title="Scheduling Agent Playground", layout="wide")
    initialize_state()

    st.title("Scheduling Agent Playground")

    with st.sidebar:
        st.header("Agent Settings")
        use_ollama = st.toggle("Use local Ollama", value=False)
        model = st.text_input("Model", value=agent.MODEL)
        ollama_url = st.text_input("Ollama URL", value=agent.OLLAMA_URL)
        st.divider()
        if st.button("Load Demo Request", width="stretch"):
            st.session_state.request_text = read_text(DEMO_REQUEST_PATH)
        if st.button("Clear Chat", width="stretch"):
            st.session_state.messages = []
            st.session_state.last_result = None

    left, right = st.columns([0.95, 1.35], gap="large")

    with left:
        st.subheader("Request")
        st.text_area(
            "Natural language scheduling request",
            key="request_text",
            height=390,
            label_visibility="collapsed",
        )
        run_clicked = st.button("Run Agent", type="primary", width="stretch")

        st.subheader("Chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask or paste a scheduling request")
        if prompt:
            st.session_state.request_text = prompt
            run_clicked = True

    if run_clicked:
        request_text = st.session_state.request_text.strip()
        if not request_text:
            st.session_state.messages.append(
                {"role": "assistant", "content": "Please enter a scheduling request first."}
            )
        else:
            st.session_state.messages.append({"role": "user", "content": request_text})
            try:
                with st.spinner("Parsing, scheduling, comparing, and saving outputs..."):
                    result = run_request(request_text, use_ollama, model, ollama_url)
                st.session_state.last_result = result
                reply = (
                    f"I parsed {len(result['jobs'])} jobs and {len(result['machines'])} machines, "
                    f"compared {', '.join(result['request']['rules'])}, and recommend "
                    f"**{result['best_rule']}** by total lateness, then makespan. "
                    f"Results were saved to `{display_path(Path(result['run_dir']))}`."
                )
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as exc:
                st.session_state.last_result = None
                st.session_state.messages.append({"role": "assistant", "content": assistant_error_message(exc)})
            st.rerun()

    with right:
        result = st.session_state.last_result
        if result:
            render_result(result)
        else:
            st.subheader("Results")
            st.info("Run the agent to see the recommendation, comparison table, Gantt chart, and saved files.")


if __name__ == "__main__":
    main()
