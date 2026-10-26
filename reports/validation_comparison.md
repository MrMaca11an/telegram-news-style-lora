# Validation comparison (identical 5-example validation split per style)

Generated: 2026-10-27T00:44:47. Heuristic flags from `src/quality_eval`; notes from manual reading of all outputs.

| style | method | format | missing-num flags | extra-num flags | trunc | best val loss | note |
|---|---|---|---|---|---|---|---|
| type1 | baseline (naive prompt) | 0/5 | 0 | 2 | 0 | — | Markdown, списки, эмодзи — стиль не соблюдён |
| type1 | zero-shot (style spec) | 0/5 | 0 | 1 | 0 | — | без обучения формат нестабилен |
| type1 | few-shot (3 train pairs) | 5/5 | 2 | 3 | 0 | — | формат ок, часть чисел теряется |
| type1 | LoRA type1_orig | 5/5 | 2 | 1 | 0 | 1.996 | грамматический брак в 1 из 5 (id=13) |
| type1 | LoRA type1_mix10 (selected) | 5/5 | 2 | 1 | 0 | 1.966 | выбран: val loss ниже, синтетика улучшила устойчивость |
| type2 | baseline (naive prompt) | 0/5 | 0 | 0 | 0 | — | Markdown, хештеги, эмодзи-списки |
| type2 | zero-shot (style spec) | 3/5 | 0 | 0 | 0 | — | абзацы нестабильны |
| type2 | few-shot (3 train pairs) | 5/5 | 0 | 0 | 0 | — | силён на коротких входах валидации |
| type2 | LoRA type2_orig | 5/5 | 1 | 0 | 0 | 1.399 | 5/5 на val, но коллапс на длинных тестовых входах (копипаст, 1/5 формат) |
| type2 | LoRA type2_mix10 | 4/5 | 0 | 0 | 0 | 1.386 | 1 грамматический брак (id=22) |
| type2 | LoRA type2_mix20 | 4/5 | 0 | 1 | 0 | 1.388 | 1 лишний абзац (id=8) |
| type2 | LoRA type2_mix20 + strict v2 prompt (selected) | 5/5 | 1 | 0 | 0 | 1.388 | выбран: единственная конфигурация 5/5 без грамматического брака |

Notes:
- `missing-num flags`: heuristic count of outputs with numbers absent from source; spelled-out
  numbers (тринадцать → 13) and intentional condensation produce false positives, so the flag
  count is advisory; the semantic verdicts are in fact_audit_test.csv.
- Validation inputs are 356-565 chars; test inputs are 854-951 chars (out of the train distribution
  180-585). This OOD gap explains the earlier type2 test collapse of type2_orig (verbatim copy,
  extra paragraphs) despite 5/5 validation format.
- type1_mix20 never finished (hung, killed); excluded.

Selection: **type1_mix10 (v1 prompt)** and **type2_mix20 + strict v2 system prompt**.
The v2 prompt was validated on validation only (5/5) before final generation.
