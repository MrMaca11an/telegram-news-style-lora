"""Display input and all three generated methods for side-by-side manual scoring."""
import argparse
from src.common import ROOT,read_jsonl

def main():
    a=argparse.ArgumentParser();a.add_argument('--style',required=True,choices=['type1','type2']);a.add_argument('--split',default='validation',choices=['validation','test']);args=a.parse_args()
    files={m:ROOT/f'outputs/{args.style}_{m}_{args.split}.jsonl' for m in ['baseline','fewshot','lora']}
    for m,path in files.items():
        if not path.exists():raise SystemExit('Missing '+str(path))
    results={m:{str(r['id']):r for r in read_jsonl(f)} for m,f in files.items()}
    for ident in results['baseline']:
        print('\n'+'='*75,'\nid:',ident)
        print('SOURCE:\n',results['baseline'][ident]['input'])
        for method in results:
            print(f'\n{method.upper()}:\n',results[method][ident]['output'])

if __name__=='__main__':main()
