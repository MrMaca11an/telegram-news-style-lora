"""Export 5 test outputs for each style after choosing best method on validation."""
import argparse
from src.common import ROOT,read_jsonl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant',default='orig')
    ap.add_argument('--method-type1',default='lora',choices=['baseline','zeroshot','fewshot','lora'])
    ap.add_argument('--method-type2',default='lora',choices=['baseline','zeroshot','fewshot','lora'])
    ap.add_argument('--run-type1',default='type1_orig')
    ap.add_argument('--run-type2',default='type2_orig')
    a=ap.parse_args()
    for style,method,run in [('type1',a.method_type1,a.run_type1),('type2',a.method_type2,a.run_type2)]:
        tag=f'{style}_{method}' if method!='lora' else f'{style}_lora_{run}'
        src=ROOT/'outputs'/a.variant/f'{tag}_test.jsonl'
        if not src.is_file():raise SystemExit('Missing '+str(src))
        rows=read_jsonl(src)
        if len(rows)!=5 or [str(r['id']) for r in rows]!=['1','2','3','4','5']:
            raise RuntimeError('Expected 5 ordered test inputs: '+str(src))
        out=ROOT/f'outputs_{style}.txt'
        out.write_text(('\n\n'+'='*42+'\n\n').join(x['output'] for x in rows)+'\n',encoding='utf8')
        print('wrote',out)

if __name__=='__main__':main()
