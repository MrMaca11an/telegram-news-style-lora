"""Diagnostics: format, token-limit and numeric warnings. Not a semantic fact checker."""
import argparse, json
from pathlib import Path
from src.common import ROOT,output_checks,read_jsonl,numbers

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--style',required=True,choices=['type1','type2'])
    ap.add_argument('--predictions',required=True)
    ap.add_argument('--save-report',default='')
    a=ap.parse_args()
    rows=read_jsonl(a.predictions)
    details=[]
    for r in rows:
        d=output_checks(a.style,r['output'])
        d.update(id=r['id'],extra_numbers_review=sorted(numbers(r['output'])-numbers(r['input'])),
                 hit_token_limit=r.get('hit_token_limit'),generated_tokens=r.get('generated_tokens'))
        details.append(d)
        print(json.dumps(d,ensure_ascii=False))
    summary={'style':a.style,'file':a.predictions,'n':len(details),
      'format_passed':sum(x['pass'] for x in details),
      'hit_token_limit':sum(x['hit_token_limit'] is True for x in details),
      'number_flags':sum(bool(x['extra_numbers_review']) for x in details),
      'note':'format is not content quality. Manually verify attribution, dates, status and factual accuracy.'}
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if a.save_report:
        Path(a.save_report).parent.mkdir(parents=True,exist_ok=True)
        Path(a.save_report).write_text(json.dumps({'summary':summary,'details':details},ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':main()
