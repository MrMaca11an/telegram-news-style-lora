"""Quality evaluation: structural style checks, normalized numeric audit,
and comparison tables across methods on identical validation examples.
Heuristic layer only: it flags, it does not certify semantic truth.
"""
import argparse, json, re, unicodedata
from pathlib import Path
from src.common import ROOT, SIGNATURE, output_checks, read_jsonl, write_jsonl

# Emoji including ZWJ sequences and variation selectors, anchored at string start.
EMOJI_START = re.compile(
    r'^(?:[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D\U0001F1E6-\U0001F1FF]+)')

def structural_checks(style, output):
    s = output.strip()
    base = output_checks(style, s)
    # Emoji grapheme check that tolerates ZWJ/variation selectors but rejects bare text
    m = EMOJI_START.match(s)
    emoji_ok = bool(m) and any(unicodedata.category(ch).startswith('So') or
                               unicodedata.category(ch) == 'Sk' or
                               ('\U0001F000' <= ch <= '\U0001FAFF') for ch in m.group(0))
    checks = {
        'pass': base['pass'],
        'signature_end': base['signature'],
        'signature_inline': base.get('signature_inline', False),
        'starts_emoji_grapheme': emoji_ok if style == 'type2' else None,
        'paragraphs': base['paragraphs'],
        'markdown_free': base['markdown_free'],
        'no_service_lines': not re.search(r'^(вот|Вот|---|\*\*\*|id=|=)', s),
        'length_chars': base['length'],
    }
    return checks

_NUM_RE = re.compile(r'(?<![\w.,])(?:\d[\d ]*\d|\d)(?:[.,]\d+)?(?![\w])', re.UNICODE)

def _norm_num(tok):
    return tok.replace(' ', '').replace(',', '.')

def numbers_normalized(text):
    out = set()
    for m in _NUM_RE.finditer(text):
        n = _norm_num(m.group(0))
        try:
            v = float(n)
        except ValueError:
            continue
        out.add(int(v) if v == int(v) else round(v, 3))
    return out

def numeric_audit(source, output):
    """Raw extra/missing number sets PLUS a normalized view. Units/status are NOT
    verified here - that is the manual fact table's job."""
    src, outp = numbers_normalized(source), numbers_normalized(output)
    return {
        'missing_numbers': sorted(src - outp),
        'extra_numbers': sorted(outp - src),
        'missing_critical': [n for n in sorted(src - outp) if n not in (0, 1)],
    }

def eval_predictions(style, rows, train_target_lens=None):
    details = []
    for r in rows:
        st = structural_checks(style, r['output'])
        na = numeric_audit(r['input'], r['output'])
        d = {'id': r['id'], **st, **na,
             'generated_tokens': r.get('generated_tokens'),
             'hit_token_limit': r.get('hit_token_limit')}
        if train_target_lens:
            lo, hi = min(train_target_lens), max(train_target_lens)
            d['length_in_train_range'] = lo <= st['length_chars'] * 1.0 <= hi * 1.6
        details.append(d)
    return details

def summary_line(style, details):
    return {
        'n': len(details),
        'format_pass': sum(d['pass'] for d in details),
        'missing_critical_num': sum(bool(d['missing_critical']) for d in details),
        'extra_numbers': sum(bool(d['extra_numbers']) for d in details),
        'truncations': sum(d['hit_token_limit'] is True for d in details),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--style', required=True, choices=['type1', 'type2'])
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--save-json', default='')
    args = ap.parse_args()
    rows = read_jsonl(args.predictions)
    details = eval_predictions(args.style, rows)
    print(json.dumps({'label': args.label, **summary_line(args.style, details)}, ensure_ascii=False))
    for d in details:
        print(json.dumps(d, ensure_ascii=False))
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(
            {'label': args.label, 'summary': summary_line(args.style, details), 'details': details},
            ensure_ascii=False, indent=2), encoding='utf8')

if __name__ == '__main__':
    main()
