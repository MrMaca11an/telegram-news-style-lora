"""Aggregate deterministic metrics across candidate validation runs; human judgement still needed."""
import argparse, json
from pathlib import Path
from src.common import read_jsonl,output_checks,numbers

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir',default='outputs/orig')
    a=ap.parse_args()
    root=Path(a.dir)
    table=[]
    for p in sorted(root.glob('*validation.jsonl')):
        rows=read_jsonl(p)
        if not rows:continue
        style=rows[0]['style']
        table.append((p.name,len(rows),sum(output_checks(style,x['output'])['pass'] for x in rows),
                sum(bool(numbers(x['output'])-numbers(x['input'])) for x in rows),
                sum(x.get('hit_token_limit') is True for x in rows)))
    print('FILE | N | FORMAT_PASSED | NUMERIC_REVIEW | HIT_TOKEN_LIMIT')
    for x in table:print(' | '.join(map(str,x)))
    if not table:raise SystemExit('No validation files in '+str(root))

if __name__=='__main__':main()
