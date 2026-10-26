# API examples (OpenAI-compatible, vLLM server)

Server: `deploy/vllm_24gb.sh` (candidate; smoke test on 24 GB required).
Status of execution: **not executed (hardware unavailable)**.

Choose the style by the `model` field; both adapters share the same base model.

## List models

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
# expect ids: "type1", "type2" (plus the base model)
```

## type1 (analytics / finance, @banki_oil)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "type1",
    "messages": [
      {"role": "system", "content": "<SYSTEM type1 from src/common.py>"},
      {"role": "user", "content": "<нейтральный новостной исходник>"}
    ],
    "temperature": 0,
    "max_tokens": 256
  }' | python3 -c "import json,sys;print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

Response contract: one paragraph, ends with ` @banki_oil` inline, no emoji,
no Markdown.

## type2 (city news)

Same call with `"model": "type2"` and the type2 system prompt from
`src/common.py` (`SYSTEM_V2['type2']`). Response contract: one emoji at the
start of the first line, blank line, exactly one detail paragraph, no Markdown,
no signature.

## Production constraints

- `temperature: 0` (greedy) — the evaluated configuration; sampling is untested.
- `max_tokens` + prompt tokens must be ≤ 4096; enforce app-side
  (recommended input ≤ 3072, output ≤ 256 for news posts).
- Set an API key (`--api-key ...`) and a concurrency limiter in front of the
  server; `max_num_seqs=15` queues excess requests.
- Do not log user news bodies verbatim; log ids, token counts and latency only.
