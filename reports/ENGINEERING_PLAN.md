# Engineering plan (agent, 2026-10-26)

## Facts established during inventory
- Git repo was absent -> initialized, branch improve/news-style-final, snapshot commit b54e9c5.
- All 5 trained adapters have experiment.json (type1_orig, type1_mix10, type2_orig, type2_mix10, type2_mix20).
  type1_mix20 was killed mid-run (no experiment.json) -> not usable, documented as failed.
- Data intact: 25+25 train pairs, 5 test inputs (sha256 recorded in worklog).
- Validation LoRA outputs exist for 4 adapters; type2_mix20 validation missing -> generating.
- Existing final test outputs: type1 5/5 format (1 numeric flag), type2 1/5 format.
  Type2 failure mode inspected: 4/5 outputs have 3 paragraphs instead of 2; id=3 is near-verbatim
  source copy + invented 'до 100%' + invented 'Москвы и области'. This is a content+format failure
  of the type2_orig adapter under greedy decoding with max_new_tokens=512.

## Plan
1. Generate type2_mix20 validation outputs (missing cell of the comparison matrix).
2. Fix src/train.py token-length accounting bug (BatchEncoding vs list); add tests for label
   masking (completion-only) at tokenizer level; keep all existing adapters untouched.
3. Build src/quality_eval.py: structural checks (exact signature position, emoji grapheme start,
   paragraph count, markdown-free, length vs train target distribution), normalized numeric audit
   (1,5/1.5, ranges), fact-span extraction scaffold; emit reports/validation_comparison.{csv,md}.
4. Manually review ALL validation outputs (4 adapters x 5 + baselines/zeroshot/fewshot already on
   disk) -> choose one adapter per style on validation ONLY (no test tuning).
5. Regenerate the 5 test inputs for both styles with chosen adapters, max_new_tokens=256
   (motivated by train target length distribution, not by test results), do_sample=False.
6. Deterministic lossless postprocess where needed; separate raw vs formatted; manual fact audit
   of all 10 -> reports/fact_audit_test.csv, reports/final_outputs_audit.md.
7. Production section: vLLM command candidate + memory math for hybrid Qwen3.5-2B + deploy/ scripts;
   smoke/load marked NOT EXECUTED (no 24GB GPU available).
8. Rebuild submission_repo (outputs_type1.txt, outputs_type2.txt, RESULTS.md with 2-3 real
   before/after pairs, README, tests, requirements), secrets scan, unit tests, final zip.
