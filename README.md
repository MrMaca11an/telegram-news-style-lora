# Telegram news style adaptation

Adapts neutral Russian news into two Telegram post styles with two LoRA adapters on `unsloth/Qwen3.5-2B` (BF16):

- **type1** — analytics/finance voice inspired by `@banki_oil`: one dense paragraph, signature `@banki_oil` inline at the end, no emoji/Markdown;
- **type2** — city-news voice inspired by `@moscowach`: one emoji + one-line lead, blank line, one detail paragraph, no Markdown/signature.

Full results, measured metrics and honest limitations: **[RESULTS.md](RESULTS.md)**.

## Repository layout

```
src/            pipeline code (prepare/train/generate/evaluate/quality_eval/finalize/benchmark/demo)
data/raw/       original task data (25+25 train pairs, 5 test inputs)
data/processed/ train/val splits + manifests
adapters/       selected LoRA adapters: type1 (= artifacts/type1_mix10), type2 (= artifacts/type2_mix20)
outputs/        delivered finals + raw test outputs with provenance
validation/     baseline and selected-config outputs behind the before/after pairs in RESULTS.md
reports/        validation comparison, fact audit, final audit, benchmark, worklog, FINAL_QA
deploy/         vLLM production candidate: launch script, memory math, smoke/load tests, API examples
tests/          CPU unit tests (label masking, format checks, postprocess)
```

see `BUILD_MANIFEST.json`.

## Environment

Reference environment (training): WSL2 Ubuntu 24.04, Python 3.12, CUDA GPU with ≥8 GB VRAM (BF16).
Exact versions are frozen in `requirements.lock`; `requirements-min.txt` lists the minimal set.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements-min.txt
```

A CUDA GPU with BF16 support is required for training and generation; the unit tests run on CPU.

## Reproduce

```bash
# 0. unit tests (CPU)
python -m unittest discover -s tests -v

# 1. build train/val splits (recreates data/processed; deterministic)
python -m src.prepare

# 2. train an adapter (example: the selected type1 adapter)
python -m src.train --style type1 --variant mix10 --run-name type1_mix10
python -m src.train --style type2 --variant mix20 --run-name type2_mix20

# 3. generate the 5 test posts per style (greedy, max_new_tokens=256)
python -m src.generate --style type1 --method lora --adapter-path adapters/type1 --run-name type1_mix10 --split test --new-tokens 256
python -m src.generate --style type2 --method lora --adapter-path adapters/type2 --run-name type2_mix20 --split test --new-tokens 256 --system v2

# ...or run an arbitrary news text through a selected adapter (GPU):
python -m src.demo --style type2 --text "Ваш нейтральный текст новости..."   # uses adapters/type2

# 4. quality checks + reports
python -m src.quality_eval --style type1 --predictions outputs/orig/type1_lora_type1_mix10_test.jsonl
python -m src.finalize_outputs       # applies documented editorial corrections -> outputs/final
python -m src.build_reports          # writes reports/validation_comparison.* , fact_audit_test.csv, final_outputs_audit.md

# 5. inference benchmark (one style per process — see note in src/benchmark.py)
python -m src.benchmark --styles type1
python -m src.benchmark --styles type2
```

Chat template contract: `tokenizer.apply_chat_template(..., enable_thinking=False)` everywhere; generation is greedy (`do_sample=False`), seed 42. System prompts are versioned in `src/common.py` (`SYSTEM`, `SYSTEM_V2`).

## Production (24 GB GPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r deploy/requirements-vllm.txt
./deploy/vllm_24gb.sh                 # validates env/adapters, then starts the server
python deploy/smoke_test.py           # status: NOT EXECUTED (no 24 GB hardware)
```

Memory budget and OOM-protection design: `deploy/production_memory.md`. The command is a validated-prerequisites candidate, not a field-tested setup — run the smoke test first.

## Data & licenses

- Code: MIT (`LICENSE`).
- `data/raw/*` was provided by the task assignment (channel-style references); it is included for reproduction of the take-home and is not independently licensed for redistribution.
- Model: `unsloth/Qwen3.5-2B` — see the model card on Hugging Face for its license terms. Adapters are derivative works of that model.
- No secrets, tokens or caches are committed; CI-friendly checks: `git diff --check`, `grep -rE 'hf_[A-Za-z0-9]{20,}' .` should find nothing. CPU unit tests run in GitHub Actions (`.github/workflows/tests.yml`).

## Status

- Delivered outputs: `outputs_type1.txt`, `outputs_type2.txt` (also in `outputs/final/` with provenance).
- Published: https://github.com/MrMaca11an/telegram-news-style-lora
- vLLM smoke/load tests: `NOT EXECUTED (hardware unavailable)`.
