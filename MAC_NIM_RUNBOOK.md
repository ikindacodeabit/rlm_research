# RLM vs Vanilla on NVIDIA NIM — Mac runbook

Branch **`mac-nim-allbench`**. Runs three long-context benchmarks — **RULER-32k**,
**LongBench v1**, **LongBench v2** — across **every subset**, comparing the
**Recursive Language Model (RLM)** against the **vanilla** stuff-it-in-the-prompt
baseline. No GPU/SLURM/vLLM: everything is OpenAI-compatible calls to the hosted
NVIDIA NIM catalog, so it runs from a laptop.

## What's in the comparison

| Benchmark | Task id | Subsets | Per-subset metric |
|-----------|---------|---------|-------------------|
| RULER-32k | `ruler32k` | 13 (niah_*, vt, cwe, fwe, qa_1/2) | substring recall (RULER's recall) |
| LongBench v1 | `longbench` | 16 English | task-specific: F1 / ROUGE-L / classification / retrieval / count / code-sim |
| LongBench v2 | `longbench_v2` | 6 domains | exact A/B/C/D choice accuracy |

LongBench v1 metrics mirror THUDM/LongBench (`benchmarks/longbench_metrics.py`,
dependency-free). Using one global substring metric would make summarization
subsets read ~0 and ruin the comparison, so each subset is scored its own way.

## One-time setup

```bash
cd rlm-prajna2-git
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt   # done already if .venv exists
export NVIDIA_API_KEY=nvapi-...        # get one free at https://build.nvidia.com
```

## 1. Download data (internet; ~minutes, LongBench v2 is large)

```bash
./.venv/bin/python scripts/download_data.py            # all three -> $RLM_DATA_DIR (~/rlm_data)
# or selectively: --only longbench  /  --force to refresh
```

## 2. Run the full comparison (resumable)

```bash
export NVIDIA_API_KEY=nvapi-...
bash scripts/run_all_mac.sh
```

Defaults: **20 examples/subset**, root **`meta/llama-3.3-70b-instruct`** (vanilla
baseline *and* RLM orchestrator), sub **`meta/llama-3.1-8b-instruct`** (RLM's cheap
recursive calls), **35 rpm** (NIM free tier). Results stream to
`results/mac_nim/` and a log to `logs/`. Interrupted? Just re-run — finished
examples in `results/mac_nim/*.jsonl` are skipped.

### Knobs (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `LIMIT` | `20` | examples **per subset** |
| `TASKS` | `ruler32k longbench longbench_v2` | space-separated subset of tasks |
| `RPM` | `35` | requests/min cap — **the main speed lever** if your NIM tier allows more |
| `ROOT_MODEL` / `SUB_MODEL` | llama-3.3-70b / llama-3.1-8b | any NIM model id |
| `MAX_STEPS` | `30` | RLM code turns per example |
| `VANILLA_CHAR_LIMIT` | `300000` | ~90k tokens; keeps vanilla inside Llama-70B's 128k window. Longer contexts are truncated for **vanilla only** — RLM reads the full doc from the REPL. |

Quick smoke first: `LIMIT=5 bash scripts/run_all_mac.sh`.

## 3. Read the results

`scripts/run_all_mac.sh` ends by printing the table and writing
`results/mac_nim/comparison.csv`. Re-aggregate anytime:

```bash
./.venv/bin/python benchmarks/score.py results/mac_nim --csv results/mac_nim/comparison.csv
```

One row per **(task, subset, mode, model)**. Key columns: `acc%` (mean of the
subset's metric ×100 — recall/F1/ROUGE/etc.), `tok/q` (tokens per example),
`s/q` (latency), `unfin`/`err`. Compare `mode=vanilla` vs `mode=rlm` row-by-row.
Full RLM transcripts land in `results/mac_nim/transcripts/` for post-mortems.

## Notes

- **Scoring is metric-aware**: examples carry a `metric` tag; the runner applies
  it (`benchmarks/run_benchmark.py`). RULER keeps substring recall (faithful to
  RULER); LongBench v1/v2 use their own metrics.
- **`acc%` is not "accuracy" for F1/ROUGE subsets** — it's the mean metric value.
  That's the standard LongBench number; it just won't hit 100.
- **Cost/time**: free-tier ~40 rpm + RLM's several calls/example means the full
  default run is on the order of hours→a day. Raise `RPM` (paid tier) or lower
  `LIMIT` to go faster. It's resumable, so it's safe to stop and continue.
- Chinese LongBench subsets are intentionally excluded (need a CN model + jieba).
