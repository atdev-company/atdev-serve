"""Accuracy gate: GSM8K (first N test problems), greedy, thinking disabled, against a running server."""
import argparse, asyncio, json, re

import aiohttp






def last_number(s):
    s = s.replace(",", "")
    m = re.findall(r"\\boxed\{([^}]*)\}", s)
    cand = m[-1] if m else s
    nums = re.findall(r"-?\d+(?:\.\d+)?", cand)
    return nums[-1] if nums else None


async def ask(session, url, q, sem):
    body = {"model": "m", "temperature": 0.0, "max_tokens": 768,
            "messages": [{"role": "user", "content": q + "\nPut the final answer in \\boxed{}."}],
            "chat_template_kwargs": {"enable_thinking": False}}
    async with sem, session.post(url, json=body) as r:
        d = await r.json()
    return d["choices"][0]["message"]["content"] or ""


async def main(a):
    import os
    rows = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "gsm8k_test.jsonl"))][: a.n]
    sem = asyncio.Semaphore(a.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=7200)) as s:
        outs = await asyncio.gather(*[ask(s, f"{a.base_url}/v1/chat/completions", q, sem) for q in [r["question"] for r in rows]])
    gold = [g.split("####")[-1].strip().replace(",", "") for g in [r["answer"] for r in rows]]
    correct = [last_number(o) is not None and float(last_number(o)) == float(g) for o, g in zip(outs, gold)]
    res = {"gsm8k_acc": sum(correct) / len(correct), "n": len(correct)}
    if a.out:
        json.dump({"summary": res, "outputs": outs}, open(a.out, "w"), ensure_ascii=False)
    print(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default="")
    asyncio.run(main(ap.parse_args()))
