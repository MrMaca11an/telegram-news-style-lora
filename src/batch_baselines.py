"""Batch base-model evaluation: load model once for both styles and three prompting methods.

Validation only. Never uses final test records for selecting a method.
"""
import argparse
import json
from pathlib import Path

from src.common import ROOT, BASE_MODEL, read_jsonl, write_jsonl, prompt_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new-tokens', type=int, default=512)
    ap.add_argument('--context-length', type=int, default=4096)
    ap.add_argument('--shots', type=int, default=3)
    ap.add_argument('--model', default=BASE_MODEL)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required')
    from unsloth import FastLanguageModel
    print('Loading base model once:', args.model, flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.context_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    FastLanguageModel.for_inference(model)
    model.eval()

    generated = 0
    for style in ('type1', 'type2'):
        rows = read_jsonl(ROOT / f'data/processed/orig/{style}_val_manifest.jsonl')
        training = read_jsonl(ROOT / f'data/processed/orig/{style}_train_manifest.jsonl')
        shots = [{'input': r['input'], 'output': r['output']}
                 for r in training if r.get('source') == 'original'][:args.shots]
        for method in ('baseline', 'zeroshot', 'fewshot'):
            predictions = []
            for row in rows:
                messages = prompt_messages(
                    style, row['input'],
                    method='baseline' if method == 'baseline' else 'style',
                    shots=shots if method == 'fewshot' else [],
                )
                enc = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors='pt', enable_thinking=False,
                )
                enc = {key: value.to('cuda:0') for key, value in enc.items()}
                input_tokens = int(enc['input_ids'].shape[1])
                if input_tokens + args.new_tokens > args.context_length:
                    raise RuntimeError(f'{style}/{method}: token context exceeds {args.context_length}')
                with torch.inference_mode():
                    seq = model.generate(
                        **enc, max_new_tokens=args.new_tokens,
                        do_sample=False,
                        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                      else tokenizer.eos_token_id),
                    )
                ids = seq[0, input_tokens:]
                output = tokenizer.decode(ids, skip_special_tokens=True).strip()
                eos_ids = {tokenizer.eos_token_id}
                extra = model.generation_config.eos_token_id
                if isinstance(extra, list):
                    eos_ids.update(extra)
                elif extra is not None:
                    eos_ids.add(extra)
                eos_ended = bool(len(ids) and int(ids[-1]) in eos_ids)
                record = {
                    'id': row['id'], 'input': row['input'], 'output': output,
                    'method': method, 'style': style, 'variant': 'orig',
                    'run_name': 'base', 'prompt_tokens': input_tokens,
                    'generated_tokens': len(ids), 'max_new_tokens': args.new_tokens,
                    'hit_token_limit': len(ids) >= args.new_tokens and not eos_ended,
                    'eos_ended': eos_ended, 'split': 'validation',
                }
                predictions.append(record)
                generated += 1
                print(style, method, 'id', row['id'], 'tokens', len(ids),
                      'hit_limit', record['hit_token_limit'], repr(output[:110]), flush=True)
            outdir = ROOT / 'outputs/orig'
            outdir.mkdir(parents=True, exist_ok=True)
            stem = outdir / f'{style}_{method}_validation'
            write_jsonl(stem.with_suffix('.jsonl'), predictions)
            stem.with_suffix('.txt').write_text(
                ('\n' + '=' * 42 + '\n').join(
                    f'[id={r["id"]}]\n{r["output"]}' for r in predictions
                ) + '\n', encoding='utf-8',
            )
            print('Saved', stem, flush=True)
    print('Completed baseline/zeroshot/fewshot in single process:', generated, 'posts', flush=True)

if __name__ == '__main__':
    main()
