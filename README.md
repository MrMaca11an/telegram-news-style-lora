# Адаптация новостных текстов под Telegram-стили

Два LoRA-адаптера на `unsloth/Qwen3.5-2B` (BF16) переписывают нейтральные русскоязычные новости под два Telegram-стиля:

- **type1** — аналитика/финансы в духе `@banki_oil`: один плотный абзац, подпись `@banki_oil` в конце того же абзаца, без эмодзи и Markdown;
- **type2** — городской новостник в духе `@moscowach`: один эмодзи + конкретный лид, пустая строка, один абзац с деталями, без Markdown и подписи.

Полные результаты, измеренные метрики и честные ограничения: **[RESULTS.md](RESULTS.md)**.

## Структура репозитория

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

## Окружение

Референсное окружение (обучение): WSL2 Ubuntu 24.04, Python 3.12, CUDA GPU с ≥8 ГБ VRAM (BF16).
Точные версии зафиксированы в `requirements.lock`; минимальный набор — в `requirements-min.txt`.

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # под вашу CUDA
pip install -r requirements-min.txt
```

Для обучения и генерации нужна CUDA GPU с поддержкой BF16; юниттесты идут на CPU.

## Воспроизведение

```bash
# 0. юниттесты (CPU)
python -m unittest discover -s tests -v

# 1. собрать train/val разбиения (детерминированно)
python -m src.prepare

# 2. обучить адаптер (пример — выбранный type1)
python -m src.train --style type1 --variant mix10 --run-name type1_mix10
python -m src.train --style type2 --variant mix20 --run-name type2_mix20

# 3. сгенерировать 5 тестовых постов на каждый стиль (greedy, max_new_tokens=256)
python -m src.generate --style type1 --method lora --adapter-path adapters/type1 --run-name type1_mix10 --split test --new-tokens 256
python -m src.generate --style type2 --method lora --adapter-path adapters/type2 --run-name type2_mix20 --split test --new-tokens 256 --system v2

# ...или прогнать произвольную новость через выбранный адаптер (GPU):
python -m src.demo --style type2 --text "Ваш нейтральный текст новости..."   # uses adapters/type2

# 4. проверки качества + отчёты
python -m src.quality_eval --style type1 --predictions outputs/orig/type1_lora_type1_mix10_test.jsonl
python -m src.finalize_outputs       # документированные редакторские правки -> outputs/final
python -m src.build_reports          # reports/validation_comparison.*, fact_audit_test.csv, final_outputs_audit.md

# 5. benchmark инференса (один стиль на процесс — см. примечание в src/benchmark.py)
python -m src.benchmark --styles type1
python -m src.benchmark --styles type2
```

Контракт chat template: везде `tokenizer.apply_chat_template(..., enable_thinking=False)`; генерация жадная (`do_sample=False`), seed 42. Системные промпты версионируются в `src/common.py` (`SYSTEM`, `SYSTEM_V2`).

## Production (GPU 24 ГБ)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r deploy/requirements-vllm.txt
./deploy/vllm_24gb.sh                 # валидирует окружение и адаптеры, затем стартует сервер
python deploy/smoke_test.py           # статус: NOT EXECUTED (нет 24 ГБ GPU)
```

Бюджет памяти и защита от OOM: `deploy/production_memory.md`. Команда — кандидат с валидацией пререквизитов, а не обкатанная конфигурация: сначала smoke-тест.

## Данные и лицензии

- Код — MIT (`LICENSE`).
- `data/raw/*` предоставлены заданием (стилевые референсы каналов); включены для воспроизводимости тестового задания и не лицензированы для самостоятельного распространения.
- Модель: `unsloth/Qwen3.5-2B` — условия лицензии см. на карточке модели в Hugging Face; адаптеры являются производным произведением этой модели.
- Секретов, токенов и кэшей в репозитории нет; проверки: `git diff --check` и `grep -rE 'hf_[A-Za-z0-9]{20,}' .` не должны ничего находить. CPU-тесты также гоняются в GitHub Actions (`.github/workflows/tests.yml`).

## Статус

- Итоговые тексты: `outputs_type1.txt`, `outputs_type2.txt` (также в `outputs/final/` с провенансом).
- Опубликовано: https://github.com/MrMaca11an/telegram-news-style-lora
- vLLM smoke/load тесты: `NOT EXECUTED (hardware unavailable)`.
