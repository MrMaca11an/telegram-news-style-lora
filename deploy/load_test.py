"""Concurrency load test for the vLLM production server. NOT EXECUTED (no 24 GB GPU).

Run on the 24 GB host while the server from deploy/vllm_24gb.sh is up:
    pip install aiohttp
    python deploy/load_test.py --concurrency 15 --cases 512 2048 4096

Saturation matrix: 1/5/10/15 concurrent clients x combined-token cases
(512/2048/4096 prompt+completion) with 128-256 output tokens. Reports p50/p95
latency, throughput and errors. Watch nvidia-smi for OOM/crashes; on pressure
reduce --max-num-seqs / --max-num-batched-tokens, or add --enforce-eager.
"""
import argparse, asyncio, json, random, statistics, time

SYSTEMS = {
    "type1": "Ты редактор русскоязычного Telegram-канала с живой экономической и социальной повесткой.",
    "type2": "Ты редактор оперативного городского Telegram-канала.",
}
FILLER = ("Аналитики отмечают рост сегмента и сохранение спроса; участники рынка ожидают "
          "дополнительных данных в следующем квартале. ")  # ~110 chars

async def one(session, host, style, prompt_tokens_target, out_tokens):
    # crude token budgeting: ~2.5 chars per Russian token, filler inflation for long cases
    need = max(0, prompt_tokens_target - 60)
    body = TEST_BASE + " " + FILLER * int(need * 2.5 / len(FILLER))
    payload = {"model": style,
               "messages": [{"role": "system", "content": SYSTEMS[style]},
                            {"role": "user", "content": body[:int(need * 2.5)]}],
               "temperature": 0, "max_tokens": out_tokens}
    t0 = time.monotonic()
    async with session.post(host + "/v1/chat/completions", json=payload,
                            timeout=aiohttp.ClientTimeout(total=600)) as r:
        data = await r.json()
        dt = time.monotonic() - t0
        ok = r.status == 200 and data.get("choices")
        return ok, dt, data.get("usage", {})

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=15)
    ap.add_argument("--requests", type=int, default=45)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--ramp", action="store_true", help="run 1,5,10,15 ladder")
    args = ap.parse_args()
    import aiohttp
    levels = [1, 5, 10, 15] if args.ramp else [args.concurrency]
    results = {}
    async with aiohttp.ClientSession() as session:
        for c in levels:
            ok_lat, errs = [], 0
            sem = asyncio.Semaphore(c)
            async def guarded(i):
                nonlocal errs
                async with sem:
                    style = "type1" if i % 2 == 0 else "type2"
                    try:
                        ok, dt, usage = await one(session, args.host, style,
                                                  args.prompt_tokens, args.out_tokens)
                        if ok: ok_lat.append(dt)
                        else: errs += 1
                    except Exception:
                        errs += 1
            t0 = time.monotonic()
            await asyncio.gather(*[guarded(i) for i in range(args.requests)])
            wall = time.monotonic() - t0
            results[c] = {"n": args.requests, "ok": len(ok_lat), "errors": errs,
                          "wall_s": round(wall, 1),
                          "p50_s": round(statistics.median(ok_lat), 2) if ok_lat else None,
                          "p95_s": round(sorted(ok_lat)[int(len(ok_lat)*0.95)], 2) if ok_lat else None,
                          "req_per_s": round(len(ok_lat) / wall, 2) if wall else None}
            print(c, json.dumps(results[c]))
    print(json.dumps({"summary": results}, ensure_ascii=False, indent=2))

TEST_BASE = "Крупнейшие сервисы экспресс-доставки объявили о пересмотре тарифов на курьерские услуги."

if __name__ == "__main__":
    main()
