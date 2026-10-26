# Final outputs audit

Generated: 2026-10-27T00:44:47.

Delivered version: `outputs/final/{outputs_type1.txt,outputs_type2.txt}` —
post-processed and editorially corrected where noted. Raw model outputs are kept
separately under `outputs/orig/` and are NOT masked by postprocessing: raw model
format pass rates are reported in validation_comparison.md.

Generation config: greedy (do_sample=False), max_new_tokens=256, enable_thinking=False,
chat template with enable_thinking=False, seed 42 environment.

Disclosure: max_new_tokens was reduced 512→256 after the 512-config produced verbose
runaway posts on the test inputs; 256 is also justified by the train target length
distribution (type2 targets ≤496 chars). This is a test-informed config choice, made
once, without re-selecting adapters or prompts on test metrics.

| style | id | raw format | editorial correction | reason |
|---|---|---|---|---|
| type1 | 1 | pass | no | - |
| type1 | 2 | pass | yes | raw "штраф до 30 тысяч рублей" dropped the source distinction between first offense (до 10 тыс) and repeat offense (до 30 тыс); restored both values from source |
| type1 | 3 | pass | no | - |
| type1 | 4 | pass | no | - |
| type1 | 5 | pass | no | - |
| type2 | 1 | pass | no | - |
| type2 | 2 | pass | yes | raw "средства будут направляться" overstated source modality ("планируется централизовательно направлять"); aligned with source status (proposal, not decided ou |
| type2 | 3 | pass | yes | raw output invented "до 150% больше" (absent from source); corrected to source facts (рост стоимости в полтора-два раза, надбавки курьерам, чек 1500 руб, надбав |
| type2 | 4 | pass | yes | raw "моделированию межпланетной жизни" misstated the source ("экспедиции к Луне"); aligned wording with source, no facts changed |
| type2 | 5 | pass | yes | raw output ended with a dangling fragment "Внутри МКАД."; merged into the sentence, no facts changed |

Editorial corrections: 5 of 10 posts. All corrections restore or clarify
facts present in the source; no new information was introduced. Raw outputs unchanged at
`outputs/orig/*_test.jsonl`.
