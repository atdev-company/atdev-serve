"""Fixed-workload streaming client for an OpenAI-compatible server.

Measures per-request TTFT and decode speed (tokens after the first / time after
the first token) on a fixed real-prompt set with greedy sampling, so speculative
decoding acceptance reflects real text rather than random tokens.
"""
import argparse, asyncio, json, os, statistics, time

import aiohttp


async def one(session, url, model, prompt, max_tokens, extra):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        **extra,
    }
    t0 = time.perf_counter()
    t_first = None
    n_tokens = None
    text = []
    async with session.post(url, json=body) as r:
        r.raise_for_status()
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                n_tokens = d["usage"]["completion_tokens"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    text.append(piece)
    t_end = time.perf_counter()
    decode_s = t_end - (t_first or t_end)
    return {
        "ttft": (t_first or t_end) - t0,
        "e2e": t_end - t0,
        "tokens": n_tokens,
        "decode_s": decode_s,
        "decode_tps": (n_tokens - 1) / decode_s if n_tokens and decode_s > 0 else None,
        "text": "".join(text),
    }


async def run(args):
    prompts = [json.loads(l)["prompt"] for l in open(args.prompts)]
    extra = json.loads(args.extra) if args.extra else {}
    url = f"{args.base_url}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        # warmup with the same concurrency shape
        await asyncio.gather(*[one(s, url, args.model, prompts[i % len(prompts)], 64, extra)
                               for i in range(max(args.concurrency, 2))])
        sem = asyncio.Semaphore(args.concurrency)

        async def guarded(p):
            async with sem:
                return await one(s, url, args.model, p, args.max_tokens, extra)

        t0 = time.perf_counter(); t_start = time.time()
        res = await asyncio.gather(*[guarded(p) for p in prompts])
        wall = time.perf_counter() - t0; t_end = time.time()
    tps = [r["decode_tps"] for r in res if r["decode_tps"]]
    total_tokens = sum(r["tokens"] or 0 for r in res)
    k = int(len(tps) * 0.1)  # trimmed mean: drop 10% each side
    trimmed = sorted(tps)[k:len(tps) - k] if k else sorted(tps)
    dec_tokens = sum((r["tokens"] or 1) - 1 for r in res if r["decode_tps"])
    dec_s = sum(r["decode_s"] for r in res if r["decode_tps"])
    summary = {
        "concurrency": args.concurrency,
        "n": len(res),
        "prompts": os.path.basename(args.prompts),
        "median_decode_tps": statistics.median(tps),
        "mean_decode_tps": statistics.mean(tps),
        "trimmed_mean_decode_tps": statistics.mean(trimmed),
        "wall_decode_tps": dec_tokens / dec_s if dec_s > 0 else None,  # total decoded tokens / total decode time (c=1: pooled tok/s)
        "agg_output_tps": total_tokens / wall,
        "median_ttft": statistics.median(r["ttft"] for r in res),
        "total_tokens": total_tokens,
        "wall_s": wall,
        "t_start": t_start,
        "t_end": t_end,
    }
    out = {"summary": summary, "requests": res}
    if args.out:
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--extra", default="", help="extra JSON fields for the request body")
    ap.add_argument("--out", default="")
    asyncio.run(run(ap.parse_args()))
