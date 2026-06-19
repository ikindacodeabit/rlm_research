# Running the RLM benchmark with locally-served Qwen3-8B on Prajna

Step-by-step guide to serve **Qwen/Qwen3-8B** on a Prajna GPU node with vLLM and run the
RLM benchmark grid (niah / multikey / longbench_v2 × vanilla / rlm) against it.

The repo code + SLURM scripts are **already edited** for Qwen3-8B (see "What was already
changed" at the bottom). Everything below is cluster commands you run yourself.

## Conventions (already baked into the scripts — don't change)
| Thing | Value |
|-------|-------|
| venv | `~/venvs/rlm` |
| HF weights cache | `$HF_HOME = ~/hf_cache` |
| benchmark data | `$RLM_DATA_DIR = ~/rlm_data` |
| GPU partition | `a40` (4×A40 48 GB/node — Qwen3-8B fits easily) |
| served port | `8000` |

> Run `pip` / `hf download` / `download_data.sh` on a **LOGIN node** (it has internet).
> Compute nodes are air-gapped (`HF_HUB_OFFLINE=1`), so weights must be cached first.
> Never run the model on a login node — only `sbatch`.

---

## Step 1 — Environment (login node, one time)
```bash
cd ~/path/to/rlm-prajna2            # wherever this repo lives on Prajna

python -m venv ~/venvs/rlm         # skip if it already exists
source ~/venvs/rlm/bin/activate

pip install -r requirements.txt    # client-side deps (openai, datasets, hf_hub, tiktoken)
pip install "vllm>=0.8.5"          # inference engine — Qwen3 needs vLLM 0.8.x+
python -c "import vllm; print('vllm', vllm.__version__)"
```
If `import vllm` errors with a CUDA/driver mismatch, see **Troubleshooting**.

## Step 2 — Download Qwen3-8B weights to the cache (login node)
```bash
export HF_HOME="$HOME/hf_cache"
source ~/venvs/rlm/bin/activate
hf download Qwen/Qwen3-8B           # older CLI: huggingface-cli download Qwen/Qwen3-8B
ls "$HF_HOME/hub" | grep -i qwen3   # expect: models--Qwen--Qwen3-8B
```

## Step 3 — Download benchmark data (login node)
niah + multikey are generated in-process (no download). Only longbench_v2 needs data:
```bash
bash slurm/download_data.sh        # writes ~/rlm_data/longbench_v2.jsonl
```

## Step 4 — Smoke test (40-min GPU job)
Validates the whole path on ONE example before you spend GPU quota on the full grid.
```bash
sbatch slurm/serve_smoke.slurm
squeue --me                                  # watch it schedule / start
tail -f logs/serve-smoke.<JOBID>.out         # main log
tail -f logs/vllm.<JOBID>.log                # vLLM model load
```
**PASS criteria** in `serve-smoke.<JOBID>.out`:
- the raw completion prints `READY`
- a 1-example NIAH correctness line prints
- the `--debug` MODEL REPLY dump contains **no `<think>` blocks**

If you see `<think>` blocks, thinking-disable isn't working — see Troubleshooting.

## Step 5 — Full benchmark grid
```bash
sbatch slurm/run_eval_local.slurm
tail -f logs/rlm-local.<JOBID>.out
```
Runs {niah, multikey, longbench_v2} × {vanilla, rlm}, 50 examples each. Resumable —
re-submitting skips examples already in the JSONL.

Outputs:
- `results/<task>.<mode>.Qwen_Qwen3-8B.jsonl` — per-example records
- `results/transcripts/<task>.rlm.Qwen_Qwen3-8B/<id>.json` — full RLM transcripts

## Step 6 — Score
```bash
source ~/venvs/rlm/bin/activate
python benchmarks/score.py --help     # check exact args
python benchmarks/score.py results
```

---

## Troubleshooting
- **`import vllm` CUDA/driver error (Step 1):** the wheel must match the node's CUDA. Check a
  GPU node's `nvidia-smi`, load a matching CUDA module (`module avail cuda` / `spack load cuda`
  per the Prajna manual), and install the corresponding vLLM build.
- **`vLLM never became ready` in the job log:** read `logs/vllm.<JOBID>.log`. Usual causes —
  weights not cached (redo Step 2) or `--max-model-len` too big for the GPU (lower it).
- **OOM at model load:** in the `.slurm` file lower `--gpu-memory-utilization` to `0.85`,
  or `--max-model-len` to `16384`.
- **`<think>` blocks appear (Step 4):** confirm `--no-think` is on the
  `python benchmarks/run_benchmark.py` line and that `rlm/client.py` has the `extra_body`
  field + fallback (see below). Both must be present for thinking to be disabled.
- **Wrong partition:** run `sinfo` and confirm `a40` / qos names before submitting.
- **`hf: command not found`:** use `huggingface-cli download Qwen/Qwen3-8B`, or
  `python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B')"`.

---

## What was already changed in the repo (for reference)
You do **not** need to redo these — they're committed in the working tree:

1. **`rlm/client.py`** — added an `extra_body: dict | None = None` field to `NIMClient`;
   `chat()` now falls back to it when no per-call `extra_body` is given.
2. **`benchmarks/run_benchmark.py`** — added a `--no-think` flag that sets
   `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` on the root and sub clients.
3. **`slurm/run_eval_local.slurm`** — `MODEL=Qwen/Qwen3-8B`, `--served-model-name "$MODEL"` on
   the `vllm serve` line, `--no-think` on the benchmark line.
4. **`slurm/serve_smoke.slurm`** — `MODEL="Qwen/Qwen3-8B"`, `--no-think` on the benchmark line.

Why `--no-think`: Qwen3 emits `<think>…</think>` reasoning by default. The RLM harness parses
a single fenced code block per step and requires the FINAL answer literal to appear in real
output (`rlm/rlm.py`), so thinking output corrupts both RLM steps and answer matching.

## Still missing / not in this repo (cluster state you provide)
- The `~/venvs/rlm` venv and `vllm` install (Step 1).
- Qwen3-8B weights in `$HF_HOME` (Step 2).
- Confirming the `a40` partition/qos names match `sinfo` on your Prajna account.
