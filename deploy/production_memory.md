# Production memory budget: unsloth/Qwen3.5-2B + 2 LoRA on one 24 GB GPU

Model config verified against the published `config.json` of
`unsloth/Qwen3.5-2B` (checked 2026-10-26, snapshot `7d5dc445`):

- 24 hidden layers, `full_attention_interval=4` → **12 full-attention layers +
  12 linear-attention layers** (not 6/18 as earlier task notes assumed);
- full attention: `num_key_value_heads=2`, `head_dim=256`;
- linear attention: 16 value heads, key/value head dim 128, conv kernel 4;
- `hidden_size=2048`, `vocab_size=248320`, tied embeddings, BF16;
- weights on disk: **4.24 GiB** (single safetensors, measured).

## Per-token KV and state costs (BF16)

| component | formula | bytes/token |
|---|---|---:|
| full-attention KV | 12 layers × 2 (K,V) × 2 heads × 256 dim × 2 B | 24,576 |
| linear-attention recurrent state (per sequence, not per token) | 12 layers × 16 heads × 128 × 128 × 2 B | 6.0 MiB/seq |
| conv states | 12 layers × 4 kernel × ~8k conv dim × 2 B (approx.) | ~0.8 MiB/seq |

Note: the task brief estimated 12,288 B/token from 6 full-attention layers;
the verified config doubles that (12 layers), i.e. **24,576 B/token**.

## Worst-case concurrency: 15 sequences × 4096 tokens

- full-attention KV: 61,440 tokens × 24,576 B = **1.41 GiB**
- linear-attention recurrent states: 15 × 6.0 MiB ≈ **88 MiB**
- conv states: ≈ **11 MiB**
- LoRA buffers: 2 adapters × ~10.9 M params × 2 B ≈ **42 MiB**
- weights: **4.24 GiB**

Subtotal ≈ 5.7 GiB. Add runtime overhead that cannot be computed from config
alone — activation workspace, CUDA graphs (~1–2 GiB unless `--enforce-eager`),
temporary buffers and fragmentation — and the realistic worst case is
**~7.5–9 GiB**.

## `gpu_memory_utilization`

`24 GiB × 0.85 = 20.4 GiB` is the vLLM budget (weights + KV blocks + workspace),
leaving ~3.6 GiB for the CUDA context and host-side variability. Since the
worst-case footprint above is well below 20.4 GiB, `0.85` is safe with margin;
even `0.60` (14.4 GiB) would fit. Keep `0.85` for headroom against long-tail
batching; the KV pool ends up larger than the worst case needs, which is
harmless because `max_model_len=4096` caps per-sequence usage and
`max_num_seqs=15` caps concurrency.

## OOM protection layers

1. `--max-model-len 4096` — hard cap on input + output per sequence;
   the application must enforce `input_tokens + max_new_tokens <= 4096`
   (recommended: input ≤ 3,072, output ≤ 1,024).
2. `--max-num-seqs 15` — caps active sequences. HTTP requests beyond this are
   queued by vLLM, not rejected; add an application-side queue limit / token
   bucket / rate limiter so queue latency stays bounded.
3. `--max-num-batched-tokens 4096` — conservative chunked-prefill budget.
4. Fallbacks if a real 24 GB stress test shows pressure: lower
   `--gpu-memory-utilization` is NOT the lever for OOM (it shrinks the KV pool);
   reduce `--max-num-seqs`, reduce `--max-num-batched-tokens`, or add
   `--enforce-eager` to drop CUDA graphs, and re-measure latency.

## Not executed

No 24 GB GPU was available; every number above is derived from the verified
config and the measured safetensors size, not from a production run.
`deploy/smoke_test.py` and `deploy/load_test.py` are ready to run on
RTX 3090 / A10G; see `deploy/api_examples.md`.
