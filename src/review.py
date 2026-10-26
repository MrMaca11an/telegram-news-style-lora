"""Generate human comparison template for two candidate methods; never silently score style."""
import argparse,csv
from pathlib import Path
from src.common import read_jsonl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--a',required=True);ap.add_argument('--b',required=True)
    ap.add_argument('--out',default='reports/manual_comparison.csv')
    x=ap.parse_args()
    a=read_jsonl(x.a);b=read_jsonl(x.b)
    if [str(r['id']) for r in a]!=[str(r['id']) for r in b]:
        raise SystemExit('Examples/order do not match')
    out=Path(x.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['id','input','candidate_A','candidate_B','A_fact_0_5','B_fact_0_5',
                         'A_style_0_5','B_style_0_5','A_format_0_5','B_format_0_5','notes'])
        for aa,bb in zip(a,b):writer.writerow([aa['id'],aa['input'],aa['output'],bb['output'],'','','','','','',''])
    print('created',out)

if __name__=='__main__':main()
