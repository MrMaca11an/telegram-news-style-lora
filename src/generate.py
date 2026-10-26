"""Raw output generation for identical validation examples. No silent repairs.
Records stopping statistics and prompt/response token counts.
"""
import argparse, json
from pathlib import Path
from src.common import ROOT,BASE_MODEL,read_jsonl,write_jsonl,prompt_messages,SYSTEM,SYSTEM_V2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--style',required=True,choices=['type1','type2'])
    ap.add_argument('--method',required=True,choices=['baseline','zeroshot','fewshot','lora'])
    ap.add_argument('--variant',default='orig')
    ap.add_argument('--run-name',default='base')
    ap.add_argument('--adapter-path',default='',
                    help='explicit adapter directory; overrides artifacts/<run-name>')
    ap.add_argument('--split',choices=['validation','test'],default='validation')
    ap.add_argument('--shots',type=int,default=3)
    ap.add_argument('--new-tokens',type=int,default=256)
    ap.add_argument('--system',choices=['v1','v2'],default='v1',
                    help='v2 uses SYSTEM_V2 strict prompts where defined (type2), validated on val split')
    ap.add_argument('--model',default=BASE_MODEL)
    ap.add_argument('--context-length',type=int,default=4096)
    args=ap.parse_args()
    import torch
    if not torch.cuda.is_available():raise SystemExit('NVIDIA CUDA required')
    from unsloth import FastLanguageModel
    if args.adapter_path:
        adapter=ROOT/args.adapter_path
        if not (adapter/'adapter_config.json').is_file():raise SystemExit('No adapter at '+str(adapter))
    else:
        adapter=ROOT/'artifacts'/args.run_name
    if args.method=='lora' and not (adapter/'adapter_config.json').is_file():
        raise SystemExit('No adapter at '+str(adapter))
    model_path=str(adapter) if args.method=='lora' else args.model
    model,tok=FastLanguageModel.from_pretrained(model_name=model_path,
        max_seq_length=args.context_length,load_in_4bit=False,load_in_16bit=True,full_finetuning=False)
    FastLanguageModel.for_inference(model)
    model.eval()
    if args.split=='validation':
        rows=read_jsonl(ROOT/f'data/processed/{args.variant}/{args.style}_val_manifest.jsonl')
    else:
        rows=[{'id':r['id'],'input':r['text']} for r in read_jsonl(ROOT/'data/raw/test_inputs.jsonl')]
    shots=[]
    if args.method=='fewshot':
        orig=[r for r in read_jsonl(ROOT/f'data/processed/{args.variant}/{args.style}_train_manifest.jsonl') if r.get('source')=='original']
        shots=[{'input':r['input'],'output':r['output']} for r in orig[:args.shots]]
    predictions=[]
    for row in rows:
        if args.method=='baseline':
            msgs=prompt_messages(args.style,row['input'],method='baseline',shots=shots)
        elif args.system=='v2' and args.style in SYSTEM_V2:
            # same structure as prompt_messages but with the v2 system prompt
            msgs=[{'role':'system','content':SYSTEM_V2[args.style]}]
            for ex in shots or []:
                msgs.extend([{'role':'user','content':ex['input']},
                             {'role':'assistant','content':ex['output']}])
            msgs.append({'role':'user','content':row['input']})
        else:
            msgs=prompt_messages(args.style,row['input'],method='style',shots=shots)
        enc=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,
            return_dict=True,return_tensors='pt',enable_thinking=False)
        enc={k:v.to('cuda:0') for k,v in enc.items()}
        if int(enc['input_ids'].shape[1]) + args.new_tokens > args.context_length:
            raise RuntimeError('Prompt + generation exceeds allocated context. Reduce --shots / --new-tokens or raise --context-length.')
        with torch.inference_mode():
            seq=model.generate(**enc,max_new_tokens=args.new_tokens,do_sample=False,
                               pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        ids=seq[0,enc['input_ids'].shape[1]:]
        output=tok.decode(ids,skip_special_tokens=True).strip()
        eos_set={tok.eos_token_id}
        if hasattr(model,'generation_config') and model.generation_config.eos_token_id is not None:
            e=model.generation_config.eos_token_id
            if isinstance(e,list):eos_set.update(e)
            else:eos_set.add(e)
        eos_ended=bool(len(ids) and int(ids[-1]) in eos_set)
        record={'id':row['id'],'input':row['input'],'output':output,
           'method':args.method,'style':args.style,'variant':args.variant,'run_name':args.run_name,
           'system':args.system,
           'prompt_tokens':int(enc['input_ids'].shape[1]),'generated_tokens':int(len(ids)),
           'max_new_tokens':args.new_tokens,'hit_token_limit':len(ids)>=args.new_tokens and not eos_ended,
           'eos_ended':eos_ended,'split':args.split}
        predictions.append(record)
        print(args.style,args.method,args.run_name,row['id'],
              'tokens=',len(ids),'hit_limit=',record['hit_token_limit'],repr(output[:110]),flush=True)
    out=ROOT/'outputs'/args.variant
    out.mkdir(parents=True,exist_ok=True)
    tag=f'{args.style}_{args.method}' if args.method!='lora' else f'{args.style}_lora_{args.run_name}'
    stem=out/f'{tag}_{args.split}'
    write_jsonl(stem.with_suffix('.jsonl'),predictions)
    stem.with_suffix('.txt').write_text(('\n'+'='*42+'\n').join(
             f'[id={r["id"]}]\n{r["output"]}' for r in predictions)+'\n',encoding='utf8')
    print('saved',stem.with_suffix('.txt'),flush=True)

if __name__=='__main__':main()
