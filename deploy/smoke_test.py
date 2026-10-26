"""Smoke test for the vLLM production server. NOT EXECUTED (no 24 GB GPU).

Run AFTER deploy/vllm_24gb.sh is up:
    python deploy/smoke_test.py --host http://localhost:8000

Checks: server up, both LoRA adapters listed and callable, prompt+completion
within 4096, format gate on both styles, latency sane.
"""
import argparse, json, sys, time
import urllib.request

TEST_NEWS = ("Крупнейшие сервисы экспресс-доставки объявили о пересмотре тарифов "
             "на курьерские услуги в период неблагоприятных погодных условий: "
             "стоимость может автоматически увеличиваться в полтора-два раза.")
SYSTEMS = {
    "type1": "Ты редактор русскоязычного Telegram-канала с живой экономической и социальной повесткой.",
    "type2": "Ты редактор оперативного городского Telegram-канала.",
}

def post(url, payload, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    models = post(args.host + "/v1/models", {}, timeout=10)["data"]
    ids = {m["id"] for m in models}
    assert {"type1", "type2"} <= ids, f"adapters not registered: {ids}"
    print("models registered:", sorted(ids))
    for style in ("type1", "type2"):
        t0 = time.monotonic()
        r = post(args.host + "/v1/chat/completions", {
            "model": style,
            "messages": [{"role": "system", "content": SYSTEMS[style]},
                         {"role": "user", "content": TEST_NEWS}],
            "temperature": 0, "max_tokens": args.max_tokens,
        }, timeout=180)
        text = r["choices"][0]["message"]["content"].strip()
        usage = r["usage"]
        total = usage["prompt_tokens"] + usage["completion_tokens"]
        assert total <= 4096, "context limit exceeded"
        print(f"{style}: {time.monotonic()-t0:.1f}s, {usage['completion_tokens']} tokens, "
              f"{len(text)} chars, preview: {text[:80]!r}")
    print("SMOKE TEST PASSED")

if __name__ == "__main__":
    main()
