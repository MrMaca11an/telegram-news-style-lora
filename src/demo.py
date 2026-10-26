"""Quickstart demo: convert an arbitrary neutral news text into either style
using the selected adapter. GPU required (BF16).

    python -m src.demo --style type2 --text "Нейтральный текст новости..."
    python -m src.demo --style type1 --text "..." --system v2

This is raw model output: run src/finalize_outputs / output_checks for the
format gate, and keep the fact-audit discipline before publishing anything.
"""
import argparse
from src.common import ROOT, SYSTEM, SYSTEM_V2, output_checks

ADAPTERS = {'type1': ROOT / 'adapters' / 'type1', 'type2': ROOT / 'adapters' / 'type2'}
FALLBACK = {'type1': ROOT / 'artifacts' / 'type1_mix10', 'type2': ROOT / 'artifacts' / 'type2_mix20'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--style', required=True, choices=['type1', 'type2'])
    ap.add_argument('--text', required=True)
    ap.add_argument('--system', choices=['v1', 'v2'], default=None,
                    help='default: v2 for type2 (selected config), v1 for type1')
    ap.add_argument('--max-new-tokens', type=int, default=256)
    args = ap.parse_args()
    if args.system is None:
        args.system = 'v2' if args.style == 'type2' else 'v1'
    adapter = ADAPTERS[args.style] if ADAPTERS[args.style].exists() else FALLBACK[args.style]
    system = SYSTEM_V2.get(args.style, SYSTEM[args.style]) if args.system == 'v2' else SYSTEM[args.style]
    import torch
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(model_name=str(adapter),
        max_seq_length=4096, load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
    FastLanguageModel.for_inference(model)
    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': args.text}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt', enable_thinking=False)
    enc = {k: v.to('cuda:0') for k, v in enc.items()}
    with torch.inference_mode():
        seq = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    text = tok.decode(seq[0, enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    checks = output_checks(args.style, text)
    print(text)
    print(f'\n--- format gate: {"PASS" if checks["pass"] else "FAIL"} {checks}', file=__import__('sys').stderr)

if __name__ == '__main__':
    main()
