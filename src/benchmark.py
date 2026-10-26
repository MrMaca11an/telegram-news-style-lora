"""Inference benchmark for the selected final pipeline on the 5 test inputs.
Measures cold model+adapter load, warm generation latency, tokens/s and peak
allocated VRAM for each style, using the exact final generation config.
Results go to reports/inference_benchmark.json (updated per style).

Run ONE style per process: loading a second Unsloth model in the same process
produces contaminated timings and degraded generation (observed: missing EOS).
Usage:
  .venv/bin/python -m src.benchmark --styles type1
  .venv/bin/python -m src.benchmark --styles type2
Laptop GPU (8 GB), NOT a 24 GB production measurement.
"""
import argparse, json, time
from pathlib import Path
from src.common import ROOT, SYSTEM, SYSTEM_V2, read_jsonl

PROMPTS = {
    'type1': ('artifacts/type1_mix10', 'v1'),
    'type2': ('artifacts/type2_mix20', 'v2'),
}

def system_for(style, version):
    if version == 'v2' and style in SYSTEM_V2:
        return SYSTEM_V2[style]
    return SYSTEM[style]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--styles', required=True, choices=sorted(PROMPTS), help='one style per process')
    ap.add_argument('--new-tokens', type=int, default=256)
    ap.add_argument('--context-length', type=int, default=4096)
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()
    import torch
    from unsloth import FastLanguageModel
    style = args.styles
    adapter, version = PROMPTS[style]
    rows = read_jsonl(ROOT / 'data/raw/test_inputs.jsonl')
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    model, tok = FastLanguageModel.from_pretrained(model_name=str(ROOT / adapter),
        max_seq_length=args.context_length, load_in_4bit=False,
        load_in_16bit=True, full_finetuning=False)
    FastLanguageModel.for_inference(model)
    model.eval()
    cold_load_s = time.monotonic() - t0
    first_pass_tokens = {}
    per_pass = []
    for _ in range(args.repeats):
        t1 = time.monotonic()
        for row in rows:
            msgs = [{'role': 'system', 'content': system_for(style, version)},
                    {'role': 'user', 'content': row['text']}]
            enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors='pt', enable_thinking=False)
            enc = {k: v.to('cuda:0') for k, v in enc.items()}
            with torch.inference_mode():
                seq = model.generate(**enc, max_new_tokens=args.new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            ids = seq[0, enc['input_ids'].shape[1]:]
            text = tok.decode(ids, skip_special_tokens=True)
            if _ == 0:
                first_pass_tokens[row['id']] = {'generated_tokens': int(len(ids)), 'chars': len(text.strip())}
        per_pass.append(round(time.monotonic() - t1, 2))
    tok_per_post = {k: v['generated_tokens'] for k, v in first_pass_tokens.items()}
    n_tok = tok_per_post[list(tok_per_post)[0]]
    entry = {
        'adapter': adapter, 'system_prompt': version,
        'cold_load_seconds': round(cold_load_s, 1),
        'warm_seconds_per_pass_of_5_posts': per_pass,
        'warm_seconds_per_post_mean': round(sum(per_pass) / len(per_pass) / len(rows), 2),
        'first_pass_tokens': first_pass_tokens,
        'approx_tokens_per_second_greedy': round(
            sum(tok_per_post.values()) / (per_pass[-1]), 1),
        'peak_allocated_gib': round(torch.cuda.max_memory_allocated() / 2**30, 3),
        'peak_reserved_gib': round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }
    print(style, json.dumps(entry, ensure_ascii=False), flush=True)
    del model
    torch.cuda.empty_cache()
    out = ROOT / 'reports' / 'inference_benchmark.json'
    if out.exists():
        report = json.loads(out.read_text(encoding='utf8'))
    else:
        report = {'gpu': torch.cuda.get_device_name(0),
                  'gpu_total_vram_gib': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                  'new_tokens_limit': args.new_tokens, 'context_length': args.context_length,
                  'repeats': args.repeats, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                  'note': ('laptop 8GB GPU measurement, not a 24GB server benchmark. '
                           'One style per process: a second Unsloth load in the same process '
                           'contaminates timings and generation (observed missing EOS).'),
                  'styles': {}}
    report['styles'][style] = entry
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('saved', out, flush=True)

if __name__ == '__main__':
    main()
