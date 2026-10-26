"""Split original pairs once; add synthetic only to training, in controlled ablations."""
import argparse, random, json, hashlib
from pathlib import Path
from src.common import ROOT, read_jsonl, write_jsonl, SYSTEM, prompt_messages

def digest(text):
    return hashlib.sha256(' '.join(text.casefold().split()).encode('utf8')).hexdigest()

def prepare(seed=42, synthetic_count=0, variant='orig'):
    if synthetic_count not in (0, 10, 20):
        raise ValueError('choose synthetic_count 0, 10 or 20')
    if '/' in variant or '..' in variant or not variant:
        raise ValueError('unsafe variant name')
    out = ROOT / 'data/processed' / variant
    out.mkdir(parents=True, exist_ok=True)
    tests = read_jsonl(ROOT / 'data/raw/test_inputs.jsonl')
    assert len(tests) == 5
    test_digests={digest(x['text']) for x in tests}
    result={}
    for style in SYSTEM:
        originals=read_jsonl(ROOT / f'data/raw/train_{style}.jsonl')
        assert len(originals) == 25
        order=list(range(25))
        random.Random(seed + int(style[-1])-1).shuffle(order)
        val_idx=set(order[:5])
        tr=[{'id':i+1, 'input':r['input'],'output':r['output'],'source':'original'} for i,r in enumerate(originals) if i not in val_idx]
        va=[{'id':i+1, 'input':r['input'],'output':r['output'],'source':'original'} for i,r in enumerate(originals) if i in val_idx]
        syn=read_jsonl(ROOT / f'data/synthetic/synthetic_{style}.jsonl')[:synthetic_count]
        if any(x.get('origin') != 'synthetic_hypothetical' for x in syn):
            raise ValueError('bad synthetic origin')
        original_digests={digest(x['input']) for x in tr+va}
        protected=original_digests|test_digests
        if len(original_digests) != 25 or original_digests & test_digests:
            raise ValueError('original duplicate or test leak')
        if any(digest(x['input']) in protected for x in syn):
            raise ValueError('synthetic exact duplicate')
        train_manifest=tr+syn
        random.Random(seed+int(style[-1])*10).shuffle(train_manifest)
        def shape(xs):
            return [{'prompt':prompt_messages(style,x['input']),
                     'completion':[{'role':'assistant','content':x['output']}]} for x in xs]
        write_jsonl(out/f'{style}_train.jsonl',shape(train_manifest))
        write_jsonl(out/f'{style}_val.jsonl',shape(va))
        write_jsonl(out/f'{style}_train_manifest.jsonl',train_manifest)
        write_jsonl(out/f'{style}_val_manifest.jsonl',va)
        result[style]={'original_train':len(tr),'synthetic_train':len(syn),'validation':len(va),
                       'validation_ids':[x['id'] for x in va]}
        print(f'{variant} {style}: train={len(tr)} original + {len(syn)} synthetic, val={len(va)}')
    (out/'split_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print('final test 5 examples remains untouched')
    return result

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant',default='orig')
    ap.add_argument('--synthetic-count',type=int,choices=[0,10,20],default=0)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    prepare(a.seed,a.synthetic_count,a.variant)
