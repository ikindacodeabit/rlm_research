"""RLM benchmark runner for marimo molab (32GB GPU, 2-3 CPU cores, free tier).

Run this as a marimo notebook: `marimo edit molab/rlm_molab.py` (locally) or
upload/paste into https://molab.marimo.io and attach the GPU runtime.

Architecture: vLLM is launched as a background subprocess from inside this
SAME notebook process and serves Qwen3 on localhost:8000/v1. The notebook then
calls the existing rlm_research.rlm.client.NIMClient (unmodified) against that
local endpoint -- no external network access needed, everything lives in one
molab session. This mirrors how the repo already runs on Prajna (vLLM +
OpenAI-compatible client), just with vLLM and the client colocated instead of
on separate cluster nodes.

Model choice: Qwen3-4B-Instruct-2507 by default (fits comfortably in 32GB
alongside vLLM's KV cache at 32k context; Qwen3-8B is selectable if the run
fits). Gautam's ask was specifically "try running qwen3 if possible".
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # RLM on molab: LOFT-32k / RULER-32k with Qwen3

        1. Installs vLLM + repo deps
        2. Launches `vllm serve` in the background against the attached GPU
        3. Points the existing `rlm_research` harness (unmodified) at
           `http://localhost:8000/v1`
        4. Runs the RLM vs vanilla benchmark grid on `loft32k` / `ruler32k`
        5. Scores results

        **Before running:** attach the GPU runtime (molab toolbar -> Runtime ->
        GPU) or this will try to load Qwen3 on CPU and be unusably slow.
        """
    )
    return


@app.cell
def _():
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "vllm>=0.8.5", "openai>=1.40", "datasets>=2.20",
         "huggingface_hub", "tiktoken>=0.7"],
        check=True,
    )
    return (subprocess,)


@app.cell
def _(mo):
    model_dropdown = mo.ui.dropdown(
        options=["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-8B"],
        value="Qwen/Qwen3-4B-Instruct-2507",
        label="Root/sub model (served by vLLM)",
    )
    max_model_len = mo.ui.slider(8192, 65536, value=32768, step=8192, label="max-model-len")
    gpu_mem_util = mo.ui.slider(0.5, 0.95, value=0.85, step=0.05, label="gpu-memory-utilization")
    mo.vstack([model_dropdown, max_model_len, gpu_mem_util])
    return gpu_mem_util, max_model_len, model_dropdown


@app.cell
def _(gpu_mem_util, max_model_len, model_dropdown, subprocess, sys):
    import os
    import time
    import atexit

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf_cache"))

    vllm_proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_dropdown.value,
            "--served-model-name", model_dropdown.value,
            "--max-model-len", str(max_model_len.value),
            "--gpu-memory-utilization", str(gpu_mem_util.value),
            "--port", "8000",
        ],
        stdout=open("/tmp/vllm_stdout.log", "w"),
        stderr=subprocess.STDOUT,
    )
    atexit.register(vllm_proc.terminate)
    print(f"vLLM starting (pid={vllm_proc.pid}), serving {model_dropdown.value} "
          f"on :8000 -- tail /tmp/vllm_stdout.log for load progress")
    return REPO_ROOT, time, vllm_proc


@app.cell
def _(mo):
    wait_button = mo.ui.run_button(label="Check / wait for vLLM readiness")
    wait_button
    return (wait_button,)


@app.cell
def _(mo, time, wait_button):
    mo.stop(not wait_button.value, mo.md("Click the button once vLLM has had ~1-2 min to load weights."))

    import urllib.request
    import urllib.error

    ready = False
    for _attempt in range(30):
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3)
            ready = True
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(5)

    mo.md("✅ vLLM is ready on :8000" if ready else "❌ vLLM did not become ready — check /tmp/vllm_stdout.log")
    return


@app.cell
def _(REPO_ROOT, sys):
    sys.path.insert(0, REPO_ROOT)

    from rlm.client import NIMClient
    from rlm.rlm import RLM, MemoryBudget, vanilla_answer
    from benchmarks.datasets import TASKS

    print("Available tasks:", list(TASKS))
    return MemoryBudget, NIMClient, RLM, TASKS, vanilla_answer


@app.cell
def _(mo):
    task_dropdown = mo.ui.dropdown(
        options=["loft32k", "ruler32k", "niah", "multikey"],
        value="loft32k",
        label="Task",
    )
    limit_slider = mo.ui.slider(1, 50, value=10, label="Number of examples")
    budget_slider = mo.ui.slider(1024, 32768, value=4096, step=1024, label="RLM memory budget (tokens)")
    mo.vstack([task_dropdown, limit_slider, budget_slider])
    return budget_slider, limit_slider, task_dropdown


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run benchmark (vanilla + rlm)")
    run_button
    return (run_button,)


@app.cell
def _(
    MemoryBudget,
    NIMClient,
    RLM,
    TASKS,
    budget_slider,
    limit_slider,
    mo,
    model_dropdown,
    run_button,
    task_dropdown,
    vanilla_answer,
):
    mo.stop(not run_button.value, mo.md("Configure options above, then click Run."))

    _model = model_dropdown.value
    root = NIMClient(model=_model, base_url="http://localhost:8000/v1", api_key="not-needed", rpm=1000)
    sub = NIMClient(model=_model, base_url="http://localhost:8000/v1", api_key="not-needed", rpm=1000)
    eb = {"chat_template_kwargs": {"enable_thinking": False}}
    root.extra_body = sub.extra_body = eb

    budget = MemoryBudget(max_context_tokens=budget_slider.value)
    rlm = RLM(root_client=root, sub_client=sub, max_steps=12, budget=budget)

    results = []
    for ex in TASKS[task_dropdown.value](limit_slider.value):
        vpred = vanilla_answer(root, ex["context"], ex["question"])
        rres = rlm.run(ex["context"], ex["question"])
        results.append({
            "id": ex["id"],
            "answers": ex["answers"],
            "vanilla_pred": vpred,
            "rlm_pred": rres.answer,
            "rlm_steps": rres.steps,
            "rlm_end_reason": rres.end_reason,
            "rlm_metrics": rres.metrics,
        })
        print(f"[{ex['id']}] vanilla={vpred!r:.60} rlm={rres.answer!r:.60} "
              f"(end={rres.end_reason}, steps={rres.steps})")

    mo.md(f"Done: {len(results)} examples")
    return (results,)


@app.cell
def _(mo, results):
    def _norm(s):
        return " ".join(str(s).lower().strip().split())

    def _correct(pred, answers):
        if pred is None:
            return False
        p = _norm(pred)
        return any(_norm(a) in p for a in answers)

    _n = len(results)
    _vanilla_acc = sum(_correct(r["vanilla_pred"], r["answers"]) for r in results) / max(_n, 1)
    _rlm_acc = sum(_correct(r["rlm_pred"], r["answers"]) for r in results) / max(_n, 1)

    mo.md(f"""
    ## Results ({_n} examples)
    | mode | accuracy |
    |---|---|
    | vanilla | {_vanilla_acc:.1%} |
    | rlm | {_rlm_acc:.1%} |
    """)
    return


if __name__ == "__main__":
    app.run()
