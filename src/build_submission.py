"""Rebuild submission_repo/ as a clean, self-contained deliverable.

Idempotent: removes and recreates the staging directory. Includes the two
selected adapters (weights only, no optimizer/training_args), all pipeline
code, tests, data, reports, deploy kit, delivered outputs and docs.
"""
import json, shutil, time
from pathlib import Path
from src.common import ROOT, read_jsonl

DST = ROOT / 'submission_repo'
ADAPTERS = {'type1': 'type1_mix10', 'type2': 'type2_mix20'}
COPY_DIRS = ['src', 'tests', 'data', 'reports', 'deploy', '.github']
COPY_FILES = ['README.md', 'RESULTS.md', 'requirements-min.txt', 'requirements.lock', 'LICENSE']

STAGING_LAYOUT = """
```
src/            код пайплайна (prepare/train/generate/evaluate/quality_eval/finalize/benchmark/demo)
data/raw/       исходные данные задания (25+25 обучающих пар, 5 тестовых новостей)
data/processed/ train/val разбиения + манифесты
adapters/       выбранные LoRA-адаптеры: type1 (= artifacts/type1_mix10), type2 (= artifacts/type2_mix20)
outputs/        финальные тексты + сырые тестовые выводы с провенансом
validation/     выводы baseline и выбранной конфигурации для пар «до/после» в RESULTS.md
reports/        сравнение на validation, факт-аудит, финальный аудит, benchmark, worklog, FINAL_QA
deploy/         production-кандидат vLLM: скрипт запуска, расчёт памяти, smoke/load тесты, примеры API
tests/          CPU-юниттесты (label masking, форматные проверки, постпроцессинг)
```
"""

def main():
    ts = time.strftime('%Y-%m-%dT%H:%M:%S')
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    for d in COPY_DIRS:
        shutil.copytree(ROOT / d, DST / d,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'checkpoint-*'))
    for f in COPY_FILES:
        shutil.copy2(ROOT / f, DST / f)
    # README: describe the staging layout, not the working-repo layout
    readme = (DST / 'README.md').read_text(encoding='utf8')
    s = readme.index('<!--LAYOUT_START-->')
    e = readme.index('<!--LAYOUT_END-->') + len('<!--LAYOUT_END-->')
    readme = readme[:s] + STAGING_LAYOUT.strip() + readme[e:]
    # remove the paragraph that describes the working-repo vs staging difference
    k = readme.find('В «чистой» сборке `submission_repo/`')
    if k != -1:
        readme = readme[:k] + readme[readme.find('\n\n', k) + 2:]
    # adapt the reproduction paths: adapters live in adapters/, not artifacts/
    readme = readme.replace(
        "python -m src.generate --style type1 --method lora --run-name type1_mix10 --split test --new-tokens 256",
        "python -m src.generate --style type1 --method lora --adapter-path adapters/type1 --run-name type1_mix10 --split test --new-tokens 256")
    readme = readme.replace(
        "python -m src.generate --style type2 --method lora --run-name type2_mix20 --split test --new-tokens 256 --system v2",
        "python -m src.generate --style type2 --method lora --adapter-path adapters/type2 --run-name type2_mix20 --split test --new-tokens 256 --system v2")
    readme = readme.replace(
        "python -m src.demo --style type2 --text \"Ваш нейтральный текст новости...\"",
        "python -m src.demo --style type2 --text \"Ваш нейтральный текст новости...\"   # uses adapters/type2")
    (DST / 'README.md').write_text(readme, encoding='utf8')
    # delivered outputs + provenance
    (DST / 'outputs').mkdir()
    for f in ['outputs_type1.txt', 'outputs_type2.txt']:
        shutil.copy2(ROOT / 'outputs/final' / f, DST / f)
        shutil.copy2(ROOT / 'outputs/final' / f, DST / 'outputs' / f)
    for st in ('type1', 'type2'):
        shutil.copy2(ROOT / 'outputs/final' / f'{st}_final_test.jsonl', DST / 'outputs')
    # raw test outputs (provenance for the audit)
    (DST / 'outputs' / 'orig').mkdir()
    for f in ['type1_lora_type1_mix10_test.jsonl', 'type2_lora_type2_mix20_test.jsonl']:
        shutil.copy2(ROOT / 'outputs/orig' / f, DST / 'outputs/orig')
    # adapters: weights + configs only
    for style, run in ADAPTERS.items():
        dst = DST / 'adapters' / style
        dst.mkdir(parents=True)
        keep = ['adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json',
                'tokenizer_config.json', 'chat_template.jinja', 'experiment.json',
                'training_history.json']
        for f in keep:
            src = ROOT / 'artifacts' / run / f
            if src.exists():
                shutil.copy2(src, dst / f)
        cfg = json.loads((dst / 'adapter_config.json').read_text())
        assert str(cfg.get('base_model_name_or_path', '')).endswith('Qwen3.5-2B'), cfg
    # validation evidence for before/after claims
    val = DST / 'validation'
    val.mkdir()
    for f in ['type1_baseline_validation.jsonl', 'type1_lora_type1_mix10_validation.jsonl',
              'type2_baseline_validation.jsonl', 'type2_lora_type2_mix20_strictprompt_validation.jsonl']:
        shutil.copy2(ROOT / 'outputs/orig' / f, val / f)
    # provenance manifest
    manifest = {
        'built': ts,
        'selected_adapters': ADAPTERS,
        'base_model': 'unsloth/Qwen3.5-2B',
        'data_sha256': {
            p.name: __import__('hashlib').sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'data/raw').glob('*.jsonl'))},
        'disclosures': [
            'outputs are raw model texts plus documented editorial corrections (see reports/final_outputs_audit.md)',
            'max_new_tokens 256 chosen after observing 512-token verbosity on test (disclosed in RESULTS.md)',
            'vLLM smoke/load tests NOT EXECUTED (no 24 GB hardware)',
            'GitHub publication: https://github.com/MrMaca11an/telegram-news-style-lora',
        ],
    }
    (DST / 'BUILD_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print('submission_repo rebuilt at', ts)

if __name__ == '__main__':
    main()
