"""OPTIONAL deterministic formatter. Never equate its fixes with model quality."""
import argparse, re
from src.common import read_jsonl, write_jsonl,output_checks

EMOJI_RE=re.compile(r'^[\U0001F300-\U0001FAFF\u2600-\u27BF]')

def format_post(style, text):
    s=text.strip()
    s=re.sub(r'^#{1,6}\s+', '', s)
    s=s.replace('**','')
    if style=='type1':
        s=s.replace('@banki_oil','').strip()
        s=' '.join(x.strip() for x in s.splitlines() if x.strip())
        return s.rstrip(' .')+'. @banki_oil' if s else '@banki_oil'
    # Type2: normalise obvious textual formatting but do not alter claims.
    s=s.replace('@banki_oil','').strip()
    if not EMOJI_RE.match(s):
        s='📰 '+s
    parts=[x.strip() for x in re.split(r'\n\s*\n',s) if x.strip()]
    if len(parts)==1:
        # Only use a text-preserving split at a sentence boundary.
        m=re.search(r'[.!?]\s+',parts[0])
        if m:
            parts=[parts[0][:m.end()].strip(),parts[0][m.end():].strip()]
    if len(parts)>=2:
        return parts[0]+'\n\n'+' '.join(parts[1:])
    return s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--style',required=True,choices=['type1','type2'])
    ap.add_argument('--predictions',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    rows=read_jsonl(a.predictions)
    for r in rows:
        r['raw_output']=r['output']
        r['output']=format_post(a.style,r['output'])
        r['postprocessed']=True
    write_jsonl(a.output,rows)
    print('saved',a.output,'passed',sum(output_checks(a.style,x['output'])['pass'] for x in rows),'/',len(rows))

if __name__=='__main__':main()
