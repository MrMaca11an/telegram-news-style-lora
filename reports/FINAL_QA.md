# FINAL QA (2026-10-27)

## Gate status

| criterion | status |
|---|---|
| 5 outputs per style, ids 1..5 in order | PASS (asserted in `src/finalize_outputs.py`) |
| 0 truncations (`hit_token_limit` false everywhere) | PASS |
| 5/5 final format both styles | PASS (raw model format was already 10/10; corrections did not mask raw results) |
| 0 unsupported/contradicted critical facts after audit | PASS — 0 contradicted, 0 unsupported in delivered texts; 5 accepted omissions documented (`reports/fact_audit_test.csv`) |
| raw vs postprocessed/corrected kept separate | PASS (`outputs/orig/` = raw; `outputs/final/` + delivered `.txt` = corrected, provenance in `reports/final_outputs_audit.md`) |
| 2–3 real before/after pairs on identical inputs | PASS (RESULTS.md §3: type1 id=20, type1 id=19, type2 id=8) |
| validation/test separation | PASS with one disclosed test-informed config choice (`new_tokens` 512→256, made once, no adapter/prompt selection on test) |
| unit tests | PASS — 15/15 (`python -m unittest discover -s tests`), also inside submission_repo |
| secrets scan | PASS — no token patterns; no caches committed |
| adapters load from BF16 base | PASS — base_model_name_or_path asserted = unsloth/Qwen3.5-2B in build; adapters exercised in generation/benchmark |
| production vLLM section | PRESENT — command candidate + verified-config memory math + smoke/load scripts; execution status: NOT EXECUTED (no 24 GB GPU) |
| GitHub publication | DONE — public repo: https://github.com/MrMaca11an/telegram-news-style-lora |

## Known limitations (disclosed)

1. Validation is 5 examples per style — selection uncertainty is non-trivial; every candidate output was also read manually.
2. Numeric checks are heuristic (normalized) + manual fact table; regex sets are advisory and include false positives/negatives.
3. vLLM compatibility of LoRA on the hybrid qwen3_5 architecture is plausible (docs) but unverified by execution; fallback documented.
4. Synthetic training data is hypothetical-event text; mix variants improved val loss/format but semantic realism is not certified.
5. Exact bit-level determinism across processes is not guaranteed with Unsloth compiled kernels (greedy runs differed by small token counts across processes); per-run outputs were audited instead.

## Deliverables

- Repository (working): `/home/alex/projects/telegram-style/qwen35-quality-v2` (git branch `improve/news-style-final`)
- Clean submission folder: `submission_repo/`; archive: `telegram_style_submission_repo.zip` (~85 MB)
- Final texts: `submission_repo/outputs_type1.txt`, `submission_repo/outputs_type2.txt` (= `outputs/final/*.txt`)
- Selected adapters: `submission_repo/adapters/type1` (= artifacts/type1_mix10), `submission_repo/adapters/type2` (= artifacts/type2_mix20)
