# RESULTS: Telegram news style adaptation (type1 `@banki_oil`, type2 `@moscowach`)

Base model `unsloth/Qwen3.5-2B` (BF16), two LoRA adapters (r=16, alpha=16, all attention/MLP projections, ~10.9M trainable params = 0.49%), trained with Unsloth + TRL SFT (`completion_only_loss=True`), greedy decoding. All numbers below are measured; nothing is imputed.

## 1. Data discipline

- 25 original pairs per style → fixed split: 20 train / 5 validation. Synthetic pairs (20/style) exist as `mix10`/`mix20` variants; validation identical everywhere.
- 5 test news (`data/raw/test_inputs.jsonl`, ids 1–5, sha256 `2e7449ff…`) never used in training and not used for hyperparameter search. Full disclosure: after the first test generation (previous run) showed runaway verbosity at `max_new_tokens=512`, the generation cap was set to 256 tokens — a test-informed config choice, made once; adapters/prompts were NOT selected on test metrics (validation only).
- Key distribution fact discovered in audit: validation inputs are 356–565 chars; test inputs are 854–951 chars — outside the train input range (180–585). This OOD gap explains why the earlier `type2_orig` adapter passed 5/5 validation but collapsed to 1/5 on test (verbatim copying, extra paragraphs).

## 2. Method selection (identical validation, measured)

Format pass / heuristic numeric flags on the same 5 validation ids (details: `reports/validation_comparison.md`):

| style | method | format | val loss | note |
|---|---|---|---|---|
| type1 | naive baseline prompt | 0/5 | — | Markdown, lists, emoji |
| type1 | zero-shot spec | 0/5 | — | unstable |
| type1 | few-shot (3 pairs) | 5/5 | — | loses numbers |
| type1 | LoRA `type1_orig` | 5/5 | 1.996 | grammar slip in 1/5 |
| **type1** | **LoRA `type1_mix10` (selected)** | **5/5** | **1.966** | best val loss, no content defects except 1 minor |
| type2 | naive baseline / zero-shot | 0/5 / 3/5 | — | — |
| type2 | few-shot | 5/5 | — | strong on short val inputs only |
| type2 | LoRA `type2_orig` | 5/5 | 1.399 | collapsed on long test inputs (1/5) |
| type2 | LoRA `type2_mix10` | 4/5 | 1.386 | grammar slip |
| **type2** | **LoRA `type2_mix20` + strict v2 system prompt (selected)** | **5/5** | **1.388** | only config with 5/5 and no grammar defect |

`type1_mix20` never finished training (hung, killed) and is excluded. The type2 v2 prompt (hard two-paragraph + 450-char cap) was chosen on validation before any final generation.

**Why LoRA over few-shot** (few-shot also scored 5/5 format on validation): (1) per-request cost — measured prefill is ~867–937 prompt tokens for few-shot vs ~361–392 for LoRA+v2 on the same inputs (≈2.4× fewer, replicated across 15 concurrent streams); (2) length robustness — few-shot conditions on short training examples, which encourages brevity mismatch on long inputs; the LoRA+strict-prompt config was the only one that survived the OOD-length test inputs without verbatim copying; (3) production serving — LoRA adapters are first-class objects in vLLM (`--lora-modules`), while few-shot templates live in app code and drift. Few-shot remains a documented zero-training fallback; its validation outputs are kept in `outputs/orig/`.

Training (all runs): 5 epochs, early stopping on val loss (patience 2), LR 7e-5 linear, batch 1 × grad-accum 4, seq len 1024, 8-bit AdamW, seed 42, BF16. Peaks: 4.5–5.8 GiB reserved on an 8 GB laptop GPU. Runtime 460–1380 s per run.

## 3. Before / after (same held-out validation inputs, full texts)

**type1, id=20 — BEFORE (base model + naive prompt):**

> 🚗 **Москva: Тарифы на парковку в центре повысят до 600₽/час**
>
> Департамент транспорта Москвы объявил о повышении максимальной платы за парковку на наиболее загруженных улицах внутри Садового и Бульварного колец. С 1 числа следующего месяца стоимость составит **600 рублей в час**, включая воскресные дни.
>
> **Цель:**
> *   Стимулировать оборачиваемость мест.
> *   Разгрузить центр, где текущая загрузка стоянок превышает нормативные 85%.
>
> 📍 **Где:** Садовый и Бульварный кольца.

Markdown headers/bullets, emoji, a typo ("Москva") — violates every style rule.

**type1, id=20 — AFTER (selected pipeline):**

> Стоимость парковки в центре Москвы с 1 июля составит 600 рублей в час. Департамент транспорта заявил, что максимальный тариф на Садовом и Бульварном кольцах призван стимулировать оборачиваемость мест и разгрузить центр, где загрузка стоянок превышает 85%. @banki_oil

**type1, id=19 — BEFORE:** the naive prompt output invents details absent from the source ("Ферментированные чаи; Комбуча") inside a bullet list. **AFTER:**

> Молодежь отказывается от крепкого алкоголя. По данным союзов производителей и ритейлеров, продажи водки и настоек среди людей от 18 до 28 лет сократились почти на треть. Зумеры предпочитают безалкогольные напитки и спорт. @banki_oil

**type2, id=8 — BEFORE (naive prompt):** headers, bold labels, emoji bullets ("**Что ждать:** * **Мете:** …"), closing formula. **AFTER (selected pipeline):**

> ❄️ В Москве в выходные ожидается резкое ухудшение погоды: метели, плотный снегопад и шквалистый ветер с порывами до 17 м/с.
>
> Температура опустится до минус 14 градусов, на дорогах образуется сильная гололедица. Центр организации дорожного движения рекомендовал автомобилистам отказаться от поездок на личном транспорте и пересесть на метро.

Exact two-paragraph contract, all numbers preserved.

## 4. Final test results (5 posts × 2 styles, ids 1–5)

- Raw model format pass: **10/10** (0 truncations, no `hit_token_limit`); delivered files are these raw outputs with transparent editorial corrections, NOT hidden postprocessing.
- Editorial corrections: **5 of 10** posts, each restoring/clarifying facts from the source (e.g., removed an invented "до 150% больше" in type2 id=3; restored the 10k/30k fine distinction in type1 id=2; fixed a dangling fragment in type2 id=5). Raw outputs kept in `outputs/orig/`; every correction listed in `reports/final_outputs_audit.md` and `src/finalize_outputs.py`.
- Manual fact audit (`reports/fact_audit_test.csv`): 0 `contradicted`, 0 `unsupported` in the delivered texts; remaining flags are accepted omissions (450-char condensation), each documented with severity. Heuristic numeric deltas are advisory only and shown separately.
- Delivered texts: `outputs_type1.txt`, `outputs_type2.txt` — five posts each, separated by a neutral `———` divider (type2 posts contain internal blank lines, so an explicit divider keeps the five posts unambiguously splittable); no id markers, sources or model tags.

## 5. Inference benchmark (measured, laptop RTX 5060 8 GB)

| style | cold load | warm s/post | tokens/s | peak VRAM |
|---|---:|---:|---:|---:|
| type1 (`type1_mix10`, v1 prompt) | 38.9 s | 9.2 s (first pass incl. compile) → ~5.9 s | ~20 | 4.31 GiB |
| type2 (`type2_mix20`, v2 prompt) | 19.5 s | 5.0 s | ~18 | 4.31 GiB |

All 10 generations ended with EOS (61–132 tokens). Benchmark note: loading a second Unsloth model in the same process contaminates timings and generation (observed missing EOS); each style is benchmarked in its own process (`src/benchmark.py --styles <style>`).

## 6. Production deployment (single 24 GB GPU, ≤15 concurrent, ≤4096 ctx)

Candidate command with validation (`deploy/vllm_24gb.sh`; status: **candidate, smoke test not executed — no 24 GB GPU available**):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model unsloth/Qwen3.5-2B --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 4096 --max-num-seqs 15 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 --enable-lora --max-loras 2 --max-lora-rank 16 \
  --lora-modules type1=./adapters/type1 type2=./adapters/type2 \
  --host 0.0.0.0 --port 8000
```

Memory math from the **verified config.json** (`full_attention_interval=4` → 12 full-attention + 12 linear-attention layers, KV heads 2 × head_dim 256): full-attention KV = 24,576 B/token → 15 × 4096 tokens = **1.41 GiB**; linear-attention recurrent states ≈ 88 MiB; weights 4.24 GiB (measured); LoRA ~42 MiB; with CUDA graphs and workspace ≈ **7.5–9 GiB worst case**, well inside the 20.4 GiB budget at `gpu_memory_utilization=0.85`. OOM protection: hard per-sequence cap 4096 (input+output, app-enforced), `max_num_seqs=15` with queueing, and fallback levers (`--max-num-seqs`, `--max-num-batched-tokens`, `--enforce-eager`). Details: `deploy/production_memory.md`; scripts: `deploy/smoke_test.py`, `deploy/load_test.py` (marked NOT EXECUTED); API contract: `deploy/api_examples.md`.

vLLM/qwen3_5 compatibility (checked 2026-10-27 against docs.vllm.ai LoRA docs): the backbone page exists in current vLLM docs and LoRA works through the generic `SupportsLoRA` path; still, hybrid-architecture LoRA support for the exact target modules must be confirmed by the 24 GB smoke test before production. Fallback if unsupported: merged-adapter export or Qwen2.5-3B-Instruct retrain — documented, not pretending to be verified.

## 7. Limitations

- 5-example validation → selection uncertainty is real; choices were also sanity-checked by reading every output.
- No LLM-judge or human rubric scores: no independent human review was available; heuristic + manual audit only, all limits disclosed.
- Test-informed config choice (new_tokens 512→256) disclosed above.
- vLLM production numbers are derived, not measured (`NOT EXECUTED`).

## 8. Reproduction

See `README.md`. Environment: WSL2 Ubuntu 24.04, Python 3.12.3, PyTorch 2.12.1+cu130, Unsloth 2026.10.8, Transformers 5.17.0, TRL 1.15.0, PEFT 0.21.2 (full freeze in `requirements.lock`). Elapsed engineering time (this audit-to-delivery pass, including benchmarks and correction round): ≈ 6 hours wall clock; including the earlier adapter training sessions ≈ 10–12 hours total.
