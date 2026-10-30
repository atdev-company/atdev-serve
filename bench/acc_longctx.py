"""Long-context sanity gate (recurrent-state fidelity): passkey + multi-key retrieval
in ~2k / 4k / 6.5k-token haystacks of natural text (GSM8K questions), greedy,
thinking disabled, against a running server. Deterministic prompts (seeded).

Prints {"longctx_acc": ..., "passkey_acc": ..., "multikey_acc": ..., "n": ...}.
"""
import argparse
import asyncio
import json
import os
import random
import re

import aiohttp

HERE = os.path.dirname(__file__)


def build(seed=0):
    rng = random.Random(seed)
    filler = [json.loads(line)["question"] for line in open(os.path.join(HERE, "gsm8k_test.jsonl"))]
    rng.shuffle(filler)
    cities = ["Lisbon", "Osaka", "Nairobi", "Lima", "Oslo", "Hanoi", "Quito", "Perth",
              "Tunis", "Riga", "Dakar", "Busan", "Cusco", "Malmo", "Porto", "Sapporo"]
    items = []
    fi = 0
    for n_chars in (8000, 16000, 26000):  # ~2k / 4k / 6.5k tokens
        for depth in (0.1, 0.5, 0.9):
            for rep in range(2):
                parts, total = [], 0
                while total < n_chars:
                    q = filler[fi % len(filler)]
                    fi += 1
                    parts.append(q)
                    total += len(q) + 1
                # passkey
                key = rng.randint(100000, 999999)
                pk = parts[:]
                pk.insert(int(len(pk) * depth), f"IMPORTANT: the secret passkey is {key}. Remember it.")
                items.append(("passkey", "\n".join(pk) +
                              "\n\nWhat is the secret passkey mentioned above? Answer with the number only.",
                              [str(key)]))
                # multi-key: 6 facts spread from `depth` onwards, ask for two of them
                cs = rng.sample(cities, 6)
                codes = {c: rng.randint(1000, 9999) for c in cs}
                mk = parts[:]
                span = len(mk)
                for j, c in enumerate(cs):
                    pos = int(span * (depth * (1 - j / 6) + (j / 6) * 0.95))
                    mk.insert(min(pos, len(mk)), f"Note: the access code for {c} is {codes[c]}.")
                ask = rng.sample(cs, 2)
                items.append(("multikey", "\n".join(mk) +
                              f"\n\nWhat are the access codes for {ask[0]} and {ask[1]}? "
                              "Answer as '<city>: <code>' lines only.",
                              [str(codes[ask[0]]), str(codes[ask[1]])]))
    return items


async def ask(session, url, prompt, sem):
    body = {"model": "m", "temperature": 0.0, "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    async with sem, session.post(url, json=body) as r:
        d = await r.json()
    if "choices" not in d:
        return "ERROR " + json.dumps(d)[:200]
    return d["choices"][0]["message"]["content"] or ""


async def main(a):
    items = build(a.seed)
    sem = asyncio.Semaphore(a.concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        outs = await asyncio.gather(*[ask(s, f"{a.base_url}/v1/chat/completions", p, sem)
                                      for _, p, _ in items])
    ok = [all(re.search(rf"\b{g}\b", o) for g in gold) for (_, _, gold), o in zip(items, outs)]
    kinds = [k for k, _, _ in items]
    res = {
        "longctx_acc": sum(ok) / len(ok),
        "passkey_acc": sum(o for o, k in zip(ok, kinds) if k == "passkey") / kinds.count("passkey"),
        "multikey_acc": sum(o for o, k in zip(ok, kinds) if k == "multikey") / kinds.count("multikey"),
        "n": len(ok),
    }
    if a.out:
        json.dump({"summary": res, "outputs": outs, "ok": ok}, open(a.out, "w"), ensure_ascii=False)
    print(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    asyncio.run(main(ap.parse_args()))
