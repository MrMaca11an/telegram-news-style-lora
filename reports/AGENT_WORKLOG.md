# Agent worklog (2026-10-26 → 2026-10-27)

All work under `/home/alex/projects/telegram-style/qwen35-quality-v2` (WSL Ubuntu 24.04, user alex, venv Python 3.12.3 / torch 2.12.1+cu130 / Unsloth 2026.10.8). Git repo initialized at session start; branch `improve/news-style-final`; snapshot commit `b54e9c5`.

## Phase 0 — inventory
- Verified data: 25+25 train pairs, 5 test inputs; sha256 recorded in BUILD_MANIFEST/RESULTS.
- All 5 adapters present with experiment.json (type1_orig, type1_mix10, type2_orig, type2_mix10, type2_mix20); type1_mix20 killed mid-run earlier (no experiment.json) — documented, not used.
- Git init + snapshot; .gitignore excludes checkpoints/caches/zips.

## Phase 1 — audit of the type2 1/5 failure
- Read all raw test outputs: type2_orig produced 3-paragraph posts (4/5), near-verbatim copy on id=3 plus invented "до 100% больше" and "Москвы и области".
- Root cause found: test inputs are 854–951 chars vs train inputs 180–585 (validation 356–565) — OOD length shift; type2_orig selected only on validation format.
- Generated missing type2_mix20 validation cell (GPU).

## Phase 2 — code fixes and tooling
- Fixed `src/train.py` token-length accounting (BatchEncoding input_ids).
- Widened emoji range in `src/common.py:output_checks` (U+1F000+, ZWJ/variation-safe start check in quality_eval).
- Added `src/quality_eval.py` (structural checks, normalized numeric audit) and CPU unit tests: 15/15 green (label-masking prefix property, completion-not-in-prompt, emoji grapheme, numeric normalization, postprocess).
- Validated a strict v2 system prompt for type2 on validation: 4/5 → 5/5 with type2_mix20. Added `SYSTEM_V2` (versioned) and `--system v2` to generate.py.
- Note on process failures: two file-corruption incidents caused by shell quoting through wsl.exe from Windows; resolved by editing files via UNC paths. Benchmark contamination (second Unsloth load in one process → missing EOS) resolved by one-load-per-process design.

## Phase 3 — final generation and fact audit
- Adapter selection on validation only: **type1_mix10** (v1), **type2_mix20** (v2 strict).
- Final test generation: greedy, new_tokens=256, 5 ids per style, 0 truncations; raw format pass 10/10.
- `src/finalize_outputs.py`: 5 transparent editorial corrections (invented "до 150%", dangling fragment, modality "будут→планируется", "межпланетной жизни→экспедиции", fine 10k/30k distinction); raw outputs untouched.
- `src/build_reports.py`: validation_comparison.{csv,md}, fact_audit_test.csv (manual verdicts + advisory numeric deltas), final_outputs_audit.md.
- Benchmark `src/benchmark.py` (one style per process): type1 38.9s cold / ~5.9s warm per post / ~20 tok/s / 4.31 GiB peak; type2 19.5s cold / 5.0s / ~18 tok/s / 4.31 GiB peak; all EOS-ended.

## Phase 4 — production and delivery
- `deploy/`: vllm_24gb.sh (validated launch candidate), production_memory.md (verified config: 12 full-attention + 12 linear layers → 24,576 B/token KV; worst case ~7.5–9 GiB; corrected the task brief's 6/18-layer assumption), smoke_test.py, load_test.py, api_examples.md, requirements-vllm.txt — all marked NOT EXECUTED (no 24 GB GPU).
- README.md, RESULTS.md (3 real before/after pairs), requirements.lock (pip freeze).
- `src/build_submission.py`: clean submission_repo (code/tests/data/reports/deploy/adapters-weights-only/validation evidence/BUILD_MANIFEST), 15/15 tests pass inside staging, secrets scan clean, zip 85 MB.
- Old model_adapters.zip removed; run_full.sh not re-run; no adapters overwritten; GitHub push not performed (PENDING_PUBLICATION).
