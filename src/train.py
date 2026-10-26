"""BF16 LoRA for Qwen3.5-2B. Train/val split and experiments are immutable by run-id."""
import argparse, inspect, json, platform, time
from pathlib import Path
from src.common import ROOT,BASE_MODEL,read_jsonl

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--style',required=True,choices=['type1','type2'])
    ap.add_argument('--variant',default='orig')
    ap.add_argument('--run-name',required=True)
    ap.add_argument('--model',default=BASE_MODEL)
    ap.add_argument('--epochs',type=float,default=5)
    ap.add_argument('--seq-len',type=int,default=1024)
    ap.add_argument('--rank',type=int,default=16)
    ap.add_argument('--alpha',type=int,default=16)
    ap.add_argument('--lr',type=float,default=7e-5)
    ap.add_argument('--grad-accum',type=int,default=4)
    args=ap.parse_args()
    if any(z in args.run_name for z in ('/','\\','..')):raise ValueError('unsafe run name')
    out=ROOT/'artifacts'/args.run_name
    if (out/'adapter_config.json').exists():
        raise SystemExit(f'Adapter already exists, choose a new --run-name: {out}')
    data=ROOT/'data/processed'/args.variant
    train_rows=read_jsonl(data/f'{args.style}_train.jsonl')
    val_rows=read_jsonl(data/f'{args.style}_val.jsonl')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit('NVIDIA CUDA with BF16 required')
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from transformers import set_seed, EarlyStoppingCallback
    from trl import SFTConfig,SFTTrainer
    set_seed(42)
    print('BF16 LoRA (no NF4 / no CPU weight offload),',args.model, args.run_name,flush=True)
    start=time.monotonic()
    model,tokenizer=FastLanguageModel.from_pretrained(
        model_name=args.model,max_seq_length=args.seq_len,
        load_in_4bit=False,load_in_16bit=True,full_finetuning=False)
    model=FastLanguageModel.get_peft_model(
        model,r=args.rank,lora_alpha=args.alpha,lora_dropout=0,bias='none',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        use_gradient_checkpointing='unsloth',random_state=42,max_seq_length=args.seq_len)
    misplaced=[(n,str(p.device)) for n,p in model.named_parameters() if p.device.type != 'cuda']
    if misplaced:
        raise RuntimeError('non-GPU model weights: '+repr(misplaced[:6]))
    for label,rows in [('train',train_rows),('validation',val_rows)]:
        lengths=[]
        for r in rows:
            tokens=tokenizer.apply_chat_template(r['prompt']+r['completion'],tokenize=True,
                        add_generation_prompt=False,enable_thinking=False)
            # apply_chat_template may return a BatchEncoding instead of a flat id list;
            # length must be computed over input_ids, otherwise len() counts dict keys.
            if hasattr(tokens,'input_ids'):
                ids=tokens['input_ids']
                ids=ids[0] if hasattr(ids,'__len__') and len(ids) and hasattr(ids[0],'__len__') else ids
                tokens=list(ids)
            lengths.append(len(tokens))
        print(label,'max tokens',max(lengths),'over limit',sum(k>args.seq_len for k in lengths),flush=True)
        if max(lengths)>args.seq_len:
            raise SystemExit(f'Input too long: {max(lengths)} > {args.seq_len}; increase --seq-len, do not truncate targets')
    opts=dict(output_dir=str(out),max_length=args.seq_len,packing=False,completion_only_loss=True,
        per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,num_train_epochs=args.epochs,
        learning_rate=args.lr,lr_scheduler_type='linear',warmup_steps=2,
        optim='adamw_8bit',gradient_checkpointing=True,bf16=True,fp16=False,
        logging_steps=1,report_to='none',save_strategy='epoch',save_total_limit=2,
        eval_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='eval_loss',
        greater_is_better=False,seed=42,data_seed=42)
    expected={'max_length','completion_only_loss','warmup_steps','eval_strategy'}
    missing=expected-set(inspect.signature(SFTConfig).parameters)
    if missing:raise RuntimeError(f'incompatible TRL; missing {missing}')
    trainer=SFTTrainer(model=model,args=SFTConfig(**opts),processing_class=tokenizer,
        train_dataset=Dataset.from_list(train_rows),eval_dataset=Dataset.from_list(val_rows),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    first=next(iter(trainer.get_train_dataloader()))
    labels=first.get('labels')
    if labels is None or int((labels!=-100).sum())<10:
        raise RuntimeError('completion supervision missing; inspect TRL dataset/chat template')
    decoded=tokenizer.decode(labels[0][labels[0]!=-100].tolist(),skip_special_tokens=False)
    print('FIRST SUPERVISED SPAN (inspect for prompt leakage):',repr(decoded[:550]),flush=True)
    # Detect the expected answer rather than mere presence of non--100 labels.
    if 'Ты редактор русскоязычного Telegram-канала' in decoded or 'Ты редактор оперативного городского Telegram-канала' in decoded:
        raise RuntimeError('Training loss contains SYSTEM prompt tokens; inspect chat template and collator')
    if len(decoded.strip())<10:
        raise RuntimeError('supervised span suspiciously short')
    torch.cuda.reset_peak_memory_stats()
    stats=trainer.train()
    out.mkdir(parents=True,exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    (out/'training_history.json').write_text(json.dumps(trainer.state.log_history,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    summary={'run_name':args.run_name,'variant':args.variant,'style':args.style,'model':args.model,
        'n_train':len(train_rows),'n_validation':len(val_rows),'epochs':args.epochs,
        'rank':args.rank,'alpha':args.alpha,'lr':args.lr,'grad_accum':args.grad_accum,
        'seq_len':args.seq_len,'train_loss':stats.training_loss,
        'best_eval_loss':trainer.state.best_metric,'runtime_sec':round(time.monotonic()-start,2),
        'peak_allocated_gib':round(torch.cuda.max_memory_allocated()/2**30,3),
        'peak_reserved_gib':round(torch.cuda.max_memory_reserved()/2**30,3),
        'gpu':torch.cuda.get_device_name(0),'python':platform.python_version(),
        'pytorch':torch.__version__}
    (out/'experiment.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
    print('saved',out,flush=True)

if __name__=='__main__':main()
