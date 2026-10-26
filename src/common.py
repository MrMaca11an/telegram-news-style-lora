"""Shared dataset, prompting and formatting helpers. No imports of GPU packages."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = 'unsloth/Qwen3.5-2B'
SIGNATURE = '@banki_oil'
SYSTEM = {
    'type1': (
        'Ты редактор русскоязычного Telegram-канала с живой экономической и социальной повесткой. '
        'Перепиши новостной исходник одним коротким связным абзацем. '
        'Начни сразу с главного факта, далее объясни смысл, допустима лёгкая ирония, если не меняет факты. '
        'В конце единственного абзаца добавь пробел и @banki_oil (не отдельной строкой). Без эмодзи, Markdown, подзаголовков, списков и лишних пояснений. '
        'Не придумывай числа, причины, последствия, комментарии, даты или заявления. '
        'Предложения, проекты и прогнозы не представляй принятыми решениями. Верни только пост.'
    ),
    'type2': (
        'Ты редактор оперативного городского Telegram-канала. '
        'Перепиши исходник как короткую городскую новость: '
        'в первой строке ровно один подходящий эмодзи и конкретная главная новость, '
        'затем пустая строка и один небольшой поясняющий абзац. '
        'Без Markdown-заголовков, выделения звёздочками, хештегов, подписи канала и служебных комментариев. '
        'Сохрани важные цифры, имена, места, сроки, статус инициативы. '
        'Не добавляй неподтверждённых деталей. Верни только пост.'
    )
}
# V2 type2 prompt: hard two-paragraph and length caps. Selected on the 5-example
# validation split (format 5/5 with adapter type2_mix20 vs 4/5 for SYSTEM['type2']).
# Validated on validation only; the 5 final test inputs were not used in this choice.
SYSTEM_V2 = {
    'type2': (
        'Ты редактор оперативного городского Telegram-канала. '
        'Перепиши исходник как короткую городскую новость ровно в двух абзацах: '
        'первая строка - один подходящий эмодзи и конкретная главная новость одним предложением; '
        'далее пустая строка и ОДИН короткий абзац с самыми важными деталями. '
        'Весь пост не длиннее 450 знаков. Ничего после второго абзаца. '
        'Без Markdown, хештегов и подписи канала. Сохрани важные цифры, имена, места, сроки, статус инициативы. '
        'Не добавляй ничего от себя и не пересказывай источник дословно. Верни только пост.'
    )
}
BASELINE = {
    'type1': 'Кратко перепиши новость для аналитического Telegram-канала на русском языке.',
    'type2': 'Кратко перепиши новость для городского Telegram-канала на русском языке.'
}


def read_jsonl(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, items):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def prompt_messages(style, source, method='style', shots=None):
    if style not in SYSTEM:
        raise ValueError(style)
    msgs = [{'role': 'system', 'content': BASELINE[style] if method == 'baseline' else SYSTEM[style]}]
    for example in shots or []:
        msgs.extend([
            {'role': 'user', 'content': example['input']},
            {'role': 'assistant', 'content': example['output']},
        ])
    msgs.append({'role': 'user', 'content': source})
    return msgs


def output_checks(style, output):
    s = output.strip()
    ends = s.endswith(SIGNATURE)
    has_emoji = bool(re.match(r'^[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF]', s))
    body = s.rsplit(SIGNATURE, 1)[0].rstrip() if ends else s
    paragraphs = [p for p in re.split(r'\n\s*\n', body) if p.strip()]
    base = bool(s and '**' not in s and not re.match(r'^#{1,6}\s', s))
    if style == 'type1':
        passed = base and ends and len(paragraphs) == 1 and not has_emoji and not re.search(r'\n\s*@banki_oil$', s)
    else:
        passed = base and not ends and has_emoji and len(paragraphs) == 2
    return {'pass': passed, 'signature': ends, 'starts_emoji': has_emoji,
            'paragraphs': len(paragraphs), 'markdown_free': base,
            'length': len(s),
            'signature_inline': bool(ends and re.search(r'[^\n]\s+@banki_oil$', s)) if style == 'type1' else False}


def numbers(text):
    return set(re.findall(r'(?<!\w)\d+(?:[.,]\d+)?(?!\w)', text))
