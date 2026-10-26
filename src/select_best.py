"""Provisional candidate selection using ONLY deterministic validation checks.

Not a semantic or factual evaluation. Preserves all raw candidates for human review.
"""
import json
from pathlib import Path
from src.common import ROOT, read_jsonl, output_checks, numbers


def main():
    base = ROOT / 'outputs/orig'
    reports = ROOT / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    selected = {}
    summary_lines = [
        '# автоматическое сравнение и предварительный выбор',
        '',
        '**важно:** оценки ниже проверяют формат и подозрительные числа, а не истинность, стилистику или полноту фактов.',
        'финальные 10 постов необходимо вручную проверить по исходникам перед сдачей.',
        '',
    ]

    for style in ('type1', 'type2'):
        val_path = ROOT / f'data/processed/orig/{style}_val_manifest.jsonl'
        valid_ids = [str(x['id']) for x in read_jsonl(val_path)]
        candidates = []
        for path in sorted(base.glob(f'{style}_*_validation.jsonl')):
            rows = read_jsonl(path)
            if len(rows) != 5 or [str(x['id']) for x in rows] != valid_ids:
                print('SKIP mismatched validation ids:', path)
                continue
            method = str(rows[0].get('method', ''))
            run = str(rows[0].get('run_name', 'base'))
            if method not in ('baseline', 'zeroshot', 'fewshot', 'lora'):
                continue
            # Ignore files with inconsistent run-id/method/style metadata.
            if not all(x.get('method') == method and x.get('style') == style and x.get('run_name') == run
                       and x.get('split') == 'validation' for x in rows):
                continue
            if method == 'lora' and not (ROOT / 'artifacts' / run / 'adapter_config.json').is_file():
                continue
            format_count = sum(bool(output_checks(style, x['output'])['pass']) for x in rows)
            numeric = sum(bool(numbers(x['output']) - numbers(x['input'])) for x in rows)
            token_limit = sum(x.get('hit_token_limit') is True for x in rows)
            length = sum(len(x['output']) for x in rows) / len(rows)
            # Proxy score only. Higher format pass dominates; warnings break ties.
            score = 100 * format_count - 12 * numeric - 40 * token_limit
            # Prefer less complex methods on EXACT ties: no synthetic-only advantage.
            priority = {'fewshot': 3, 'zeroshot': 2, 'lora': 1, 'baseline': 0}[method]
            record = {
                'style': style, 'method': method, 'run_name': run,
                'file': str(path.relative_to(ROOT)), 'format_pass': format_count,
                'numeric_review': numeric, 'hit_token_limit': token_limit,
                'mean_chars': round(length, 1), 'proxy_score': score,
                '_tie_breaker': priority,
            }
            candidates.append(record)
        if not candidates:
            raise SystemExit('No usable validation predictions for ' + style)
        ranking = sorted(candidates,
                         key=lambda x: (x['proxy_score'], x['_tie_breaker']), reverse=True)
        for x in ranking:
            x.pop('_tie_breaker', None)
        best = ranking[0]
        selected[style] = {key: best[key] for key in ('method', 'run_name', 'file')}
        selected[style]['format_pass'] = best['format_pass']
        selected[style]['proxy_score'] = best['proxy_score']
        summary_lines.extend([
            f'## {style}', '',
            '| кандидат | формат из 5 | новые числа: проверить | обрыв | proxy |',
            '|---|---:|---:|---:|---:|',
        ])
        for x in ranking:
            summary_lines.append(
                f"| {x['method']} / {x['run_name']} | {x['format_pass']} | "
                f"{x['numeric_review']} | {x['hit_token_limit']} | {x['proxy_score']} |"
            )
        summary_lines.extend([
            '', f"предварительно выбран **{best['method']} / {best['run_name']}**.",
            'это техническая эвристика, а не доказательство лучшего стиля или достоверности.',
            '',
        ])
    path = reports / 'auto_selection.json'
    path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding='utf-8')
    (reports / 'automatic_summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')
    print(json.dumps(selected, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
