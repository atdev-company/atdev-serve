# Multi-turn and agent workloads

Agents and chat clients send the same growing conversation again on every turn. Stock vLLM re-reads most of
that history each time. The `multiturn` profiles keep the state of the conversation so far (attention KV
and the recurrent GDN state) and prefill only what is new, and they speed up the prefill that remains.

Profile names: **c** = concurrent clients (the number it is tuned for), **m** = multi-turn, **s** = single-turn (see the [README](../README.md#the-four-profiles)).

| Profile | Launch | What it is |
|---|---|---|
| **c1m** | `./serve.sh c1m` (= `./serve_c1.sh --profile multiturn`) | C1 full stack with prefix caching, 131,072-token context, chunked-prefill kernels and a start-up JIT warmup |
| **c1s** | `./serve.sh c1s` (= `./serve_c1.sh --profile single`) | the previous C1 launch: no prefix caching, 8,192-token context |
| **c32m** | `./serve.sh c32m` (= `./serve_c32.sh --profile multiturn`) | the Hadamard W4A4 C32 stack with prefix caching for 32 concurrent sessions, 16,384-token context, chunked-prefill kernels and a start-up warmup |
| **c32s** | `./serve.sh c32s` (= `./serve_c32.sh --profile single`) | the previous C32 launch: Hadamard W4A4, no prefix caching, 8,192-token context |

All numbers on this page are wall-clock times of a fixed, scripted conversation replayed against the server
(speed only; the accuracy checks are in the last section). They are **full-stack** numbers, see
[What is engine-only and what is model-side](#what-is-engine-only-and-what-is-model-side).

## How it was measured

* **Traces.** `trace v5` (C1): 2 sessions of 38 turns each; the context grows from about 5k to 111k tokens
  (4.43M prompt tokens in total, 95% of them already seen in an earlier turn). `trace v5-c32`: 32 concurrent
  sessions of 7 turns, 2k to 10.5k tokens (1.44M prompt tokens, 76% repeated). Every turn pastes about 2k tokens of new
  dialogue transcripts and asks a question; the history in each request is a recorded reference reply (at most 300 tokens),
  so every arm sees identical prompts. Output is capped per turn (cap 128 / 512 / 2048
  tokens); a lower cap makes the run prefill-bound, a higher cap decode-bound.
* **Protocol.** Cold start of the prefix cache before every run, one discarded warm-up conversation, 3 repetitions,
  median reported, all arms on the same GPU in the same chain of sessions.
* **Baselines.** The fastest stock vLLM 0.31.0 setting we could find **for each cell**, on the same RedHatAI INT4
  checkpoint, with prefix caching on (`--mamba-cache-mode align`, shared-prefix checkpoint, 16-token unit):
  C1: INT4, default attention backend, no speculative decoding ("native B"); C32: INT4 with int8 activations and
  the built-in MTP head (k=3), `TRITON_ATTN` or the default backend, whichever is faster for that cap. The original
  bf16 model with the same flags is shown for scale. Nothing was tuned for the baseline less than for atdev-serve.
* **GPU.** One A100 80GB per arm (these runs used the second A100 of our test machine, and every pair of arms
  shares the GPU). Nothing in the profiles needs a particular GPU index or a second GPU.

## C1: trace v5, 2 sessions up to 111k tokens

| Output cap per turn | c1m tok/s | stock tok/s | c1m vs stock | bf16 tok/s | c1m vs bf16 | wall time: c1m / stock / bf16 | time ratio vs stock / bf16 |
|---|---:|---:|---:|---:|---:|---|---|
| 128 | **82.2** | 34.7 | **2.37x** | 15.4 | **5.32x** | 116.0 s / 268.6 s / 593.4 s | 2.32x / 5.12x |
| 512 | **159.4** | 52.0 | **3.07x** | 20.8 | **7.65x** | 205.2 s / 623.1 s / 1,497.5 s | 3.04x / 7.30x |
| 2048 | **198.6** | 58.4 | **3.40x** | 24.1 | **8.23x** | 301.2 s / 1,136.3 s / 2,716.3 s | 3.77x / 9.02x |

The speed multiples are ratios of output tokens per second over the whole run (prefill included); the time ratios can differ because the runs emit different numbers of tokens (c1m at `max_tokens` 2048: 59,821 tokens for c1m, 66,346 for stock, 65,536 for bf16).

Medians of 3 cold repetitions (spread 0.1-0.4 s for atdev-serve, 1.2 s for stock); tok/s is generated tokens
divided by the wall time of the whole trace, prefill included. The bf16 column is one repetition on the other
A100 of the same machine. The sum of time-to-first-token over the trace is 84.3 s for atdev-serve against
130.4 s for stock; the prefix-cache hit rate is 0.95 for both, so the gain comes from faster prefill of what
is new and faster decoding, not from caching more.

KV cache: the frozen bundle holds 284,569 tokens at `--max-model-len 131072` (2.2 full-length requests); the
dequantized prefill weights take GPU memory that the previous multi-turn build left to the cache.

## C32: 32 concurrent sessions

`trace v5-c32` (the headline family for C32), medians of 3 cold repetitions:

| Output cap | c32m tok/s | stock tok/s | c32m vs stock | bf16 tok/s | c32m vs bf16 | wall time: c32m / stock / bf16 | time ratio vs stock / bf16 |
|---|---:|---:|---:|---:|---:|---|---|
| 128 | **386.3** | 245.5 | **1.57x** | 71.0 | **5.44x** | 73.4 s / 115.1 s / 399.8 s | 1.57x / 5.45x |
| 512 | **857.3** | 543.5 | **1.58x** | 175.6 | **4.88x** | 117.6 s / 184.3 s / 569.7 s | 1.57x / 4.84x |
| 2048 | **1,096.1** | 719.2 | **1.52x** | 293.5 | **3.73x** | 199.1 s / 314.0 s / 771.8 s | 1.58x / 3.88x |

The speed multiples are ratios of output tokens per second over the whole run (prefill included); the time ratios can differ because the runs emit different numbers of tokens (c32m at `max_tokens` 2048: 218,240 tokens for c32m, 225,850 for stock, 226,530 for bf16).

`trace v4` (32 sessions x 6 turns, prompts up to about 9k tokens, kept for continuity with earlier measurements; these three rows were measured on the previous, unrotated C32 bundle and are kept for reference, the v4 numbers of the Hadamard bundle are 54.6 / 78.8 s at caps 128 / 512 on GPU0):

| Output cap | stock vLLM (fastest) | atdev-serve `multiturn` | speedup |
|---|---:|---:|---:|
| 128 tokens | 82.5 s | **54.3 s** | **1.52x** |
| 512 tokens | 118.1 s | **78.7 s** | **1.50x** |
| 2048 tokens | 189.2 s | **130.5 s** | **1.45x** |

The bf16 original has room for only 185,586 KV tokens next to its weights, so with 32 sessions of 10k tokens its prefix
cache thrashes (hit rate 0.0, 6-126 preemptions); that is why its whole-trace time is 3.9-5.5x longer than atdev-serve's here (3.7-5.4x lower output tok/s).
The C32 profile uses a 4-bit prefill GEMM on rotated weights; see [Accuracy](#accuracy-of-the-multi-turn-profiles).

## Replies the server generated itself ("live" conversations)

The traces above replay recorded replies. In a real conversation the next request contains the reply the server
produced, whose state vLLM does not cache (it only caches up to the end of the prompt). With
`ATDEV_EOR_CKPT=1` the server also keeps a state checkpoint near the end of each reply, so the next turn reuses it.
**This is optional and off by default.**

Live-reply v4 C32 (each arm feeds back its own replies; outputs differ between arms, so this is not an
apples-to-apples comparison of identical text; 2 cold repetitions):

| Setting | cap 128 | cap 512 | vs fastest stock (73.0 s / 118.3 s) |
|---|---:|---:|---:|
| stock vLLM, fastest per cap | 73.0 s | 118.3 s | 1.00x |
| atdev-serve `multiturn`, checkpoint off | 50.2 s | 82.1 s | 1.45x / 1.44x |
| ... checkpoint on, spacing 32, `--gpu-memory-utilization 0.93` (what `serve_c32.sh` uses) | 48.6 s | 75.1 s | 1.50x / 1.58x |
| ... checkpoint on, spacing 16, utilization 0.90 (more reuse, more overhead) | 47.6 s | 72.0 s | 1.53x / 1.64x |

Live-reply C1 (trace v5, own replies): the checkpoint does not change cap 128 (95.1-95.3 s -> 94.8 s: the reply is
small next to the pasted content of a turn) and saves about 7% at cap 512 (219.2 s -> 204.7 s, two runs in
different sessions, so indicative only), against 218.2 s / 606.3 s for stock vLLM.

Caveats of the checkpoint:

* **Cost where it cannot help.** On the recorded-reply traces (no reply is ever reused) it costs +1.3% (cap 128) and +2.7%
  (cap 512) at C32 with spacing 32, because the state snapshots are written every 32 generated tokens and the extra
  cache blocks compete with prefix blocks.
* **Memory.** C32 needs about 3% more GPU memory for the checkpoint blocks (the script raises
  `--gpu-memory-utilization` to 0.93; leave other processes on the GPU enough room).
* **Hit granularity.** The checkpoint covers the reply up to a few tens of tokens before its end (the snapshot spacing
  plus 16 for the dropped draft-token block), and only if the next request contains the reply's token ids unchanged.
* Reuse is not byte-exact: continuing from a cached state changes floating-point rounding, as for any prefix hit.
  In the correctness checks (10 C32 and 7 C1 conversations) the cached continuation matched the cold one in 4 of 10 and
  5 of 7 cases token for token, with mean log-probability differences of 0.01-0.10 (the stack's own cached-versus-cold noise floor is
  similar without the checkpoint: 2 of 10).

## Single-request speed: replacing the single-turn launch

`multiturn` pays for its cache bookkeeping on every decode step. Replacing the old launch by it costs 3.7% at c=1
on the 96-prompt benchmark of this repository (mean decode tok/s per request, 2 repetitions in one session):

| Launch | mean decode tok/s | vs best vLLM |
|---|---:|---:|
| `serve_c1.sh --profile single` | 409.5 (409.4, 409.6) | 2.09x |
| `serve_c1.sh` (`multiturn`) | 394.4 (394.9, 393.8) | 2.01x |
| best stock vLLM (DFlash2, k=7, fp8 drafter) | 196.3 | 1.00x |

If you only ever send independent single-shot requests, `--profile single` is 3.7% faster and needs less GPU memory.

## Power and energy

Measured with NVML at 1 Hz on the same A100 during the runs above (energy from the GPU's own counter, windows
aligned to the client; +-a few seconds). C1 trace v5, 3 repetitions each:

| Output cap | atdev-serve avg / peak power | J per output token | energy per run | stock vLLM avg / peak | J per token | energy per run |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 415 W / 501 W | 5.05 | 48.2 kJ | 365 W / 503 W | 10.5 | 98.1 kJ |
| 512 | 435 W / 509 W | 2.73 | 89.4 kJ | 359 W / 505 W | 6.9 | 223.5 kJ |
| 2048 | 442 W / 500 W | 2.22 | 133.0 kJ | 356 W / 508 W | 6.1 | 405.0 kJ |

atdev-serve draws 15-25% more power on average (the GPU is busier) but finishes 2.3-3.8x sooner in wall time, so a run
uses 0.49x / 0.40x / 0.33x the energy. GPU temperature stayed at 71-73 C and the SM clock at 1410 MHz in both
(a power-cap throttle flag appeared briefly at caps 512 and 2048 for atdev-serve, clock dips to 1395 MHz). Single-request
decoding (96 prompts): `multiturn` 426-431 W, 1.20 J/token; `single` 437-439 W, 1.18 J/token; stock vLLM 328 W, 1.80 J/token.
C32 trace v5-c32, 3 repetitions each, same NVML method (the end of each client run was read from the power trace itself; checked against the
exactly known end of a C1 run), against the fastest stock setting of each cap:

| Output cap | atdev-serve avg / peak power | J per output token | energy per run | fastest stock avg / peak | J per token | energy per run |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 444 W / 503 W | 1.15 | 32.6 kJ | 441 W / 504 W | 1.79 | 50.7 kJ |
| 512 | 414 W / 509 W | 0.48 | 48.6 kJ | 409 W / 470 W | 0.76 | 75.7 kJ |
| 2048 | 387 W / 510 W | 0.35 | 76.5 kJ | 387 W / 460 W | 0.54 | 121.4 kJ |

Average power is the same as stock's within 1-2%, the runs are 1.57-1.58x shorter, so a C32 run uses 0.64x / 0.64x / 0.63x the energy.
GPU temperature peaked at 72 C (stock 70-71 C), SM clock 1410 MHz throughout.

## What is engine-only and what is model-side

Following our accounting elsewhere in this repository, results are split by whether the model files change:

* **Engine only** (same checkpoint and drafter as vLLM): prefix caching for the tree-verification and MTP stacks
  (GDN state re-basing, sliding-window drafter alignment, fine-grained hits), the start-up warmup, the tokenization cache,
  fused prefill epilogues, FlashAttention-2 and FlashInfer prefill attention, the small-chunk GDN kernel.
* **Model-side or numerics-changing** (these are part of the full stack, not of an engine-only claim): the 3-bit
  W3 overlay and the re-trained, quantized DFlash2 drafter (C1); int8 activations into the W3 GEMM and the
  dequantize-once int8 prefill GEMM (`ATDEV_W3_I8`, `ATDEV_W3_DQ`); the tree-attention head split retune
  (`ATDEV_TREE_HSPL=auto`); W4A4 activation quantization on rotated weights (C32 decode, and the 4-bit prefill GEMM `ATDEV_PF_A4`).
  FlashAttention-2 and FlashInfer differ from the stock kernel at rounding level.

The multi-turn speedups above are full-stack. **No engine-only multi-turn measurement has been made**; the
engine-only single-turn figures remain 1.45x (c=1) and 1.22x (c=32), see the README.

## Accuracy of the multi-turn profiles

GSM8K (greedy, thinking off) and the 36-item long-context retrieval gate were run on the frozen bundles with the
prefix cache on; broad scores use the same harness as [ACCURACY.md](ACCURACY.md).

| | stock INT4 | C1 `multiturn` | C32 `multiturn` (Hadamard W4A4, 4-bit prefill GEMM) |
|---|---:|---:|---:|
| GSM8K, all 1319 | 94.31 | 94.24 | 94.39 |
| GSM8K, first 300 (cache on, shared prefix / unique prefix) | - | 96.67 | 97.00 / 95.33 |
| long-context retrieval | 36/36 | 36/36 | 36/36 |
| MMLU-Pro (1,400) | 71.5 | 69.0 | 71.4 |
| KMMLU Korean (1,125) | 67.1 | 63.8 | 67.6 |
| IFEval prompt strict / loose | 83.4 / 87.1 | 81.3 / 83.7 | 81.4 / 84.6 (mean of 3 runs) |
| HumanEval / HumanEval+ | 95.1 / 90.2 | 95.1 / 90.2 | 94.5 / 89.0 |
| MBPP / MBPP+ | 91.5 / 76.7 | 88.1 / 75.4 | 89.4 / 75.9 |
| WikiText-2 perplexity | 7.299 | 8.282 | 7.365 |

* **C1** matches the single-turn "C1 fast" profile (README: MMLU-Pro 69.14, KMMLU 63.91, perplexity 8.22) within the
  noise of these sample sizes; it keeps the known W3 gap to stock INT4 (MMLU-Pro -2.5, KMMLU -3.3). The chunked-prefill
  kernels add no measurable loss.
* **C32** is in the accuracy class of the Hadamard W4A4 launch (README: MMLU-Pro 72.79, KMMLU 65.07): the weights are the
  same rotated sidecar. The 4-bit prefill GEMM works on the rotated activations (rotation fused with the activation
  quantization, activation clip 0.9, layer 0 kept on 8-bit activations). An earlier bundle without the rotation lost
  perplexity (7.96) and 2-4 points on MMLU-Pro, KMMLU and HumanEval+. IFEval prompt strict is 81.4 (mean of 3 runs: 80.2 / 82.3 /
  81.7; loose 84.6), 2.0 points under stock INT4 (83.4); the run-to-run spread is up to about 4 points. `ATDEV_PF_A4=0` sends prefill through the 8-bit path instead (slower; not
  re-measured on the rotated bundle).
* Answer text under prefix caching is not byte-identical to the cache-off text (prefill is split at cache-block edges,
  which changes floating-point rounding); in-trace GSM8K-300 through the cached path stays within noise of the cache-off result (C32).

## Where the numbers come from

* **Speed.** The C1 and C32 multi-turn timings and the energy tables were measured on the frozen research bundles of our development tree (the runtime the profiles were built from). The release tree in this repository was then checked against them: C1 `multiturn`, v5 caps 128 / 512:
  115.0 / 204.4 s against 116.0 / 205.2 s, with the generated text of all 76 turns identical to the frozen run; C32 `multiturn`, v5-c32 caps 128 / 512:
  73.5 / 118.0 s against 73.4 / 117.6 s (frozen, same 3-rep family).
* **Accuracy.** The C1 broad evaluation and all C32 broad scores (MMLU-Pro, KMMLU, IFEval, HumanEval, MBPP, GSM8K-1319, long-context) were run on the
  research runtime, not on this tree. On this tree only a subset was re-run: C1 GSM8K-300 0.963; C32 WikiText-2 perplexity 7.365 (the research runtime: 7.365),
  GSM8K-300 0.963; the single-turn launches `serve_c1.sh --profile single` / `serve_c32.sh --profile single`: GSM8K-300 0.963 / 0.957, perplexity 7.209
  (the published 7.209). The C32 broad scores should be read as the research runtime's, with the release tree matching on perplexity and GSM8K-300.
* **Full stack.** Every multi-turn number on this page, speed and accuracy, is for the full stack. C32 `multiturn` uses the Hadamard-rotated weights
  (a model-side file) and C1 the W3 overlay and re-trained drafter, so none of it is an engine-only claim.

## Caveats

* **bf16 KV cache only.** The fast path stores the KV cache in bf16. An int8 KV cache kernel exists but is parked: its
  outputs are corrupted after a prefix-cache hit (stock vLLM's int8 KV shows the same), and capacity does not bind for
  these traces.
* **Memory.** C1 `multiturn` keeps a dequantized int8 copy of the W3 weights for large prefill chunks, which leaves
  284,569 KV tokens at 131,072 context. Longer contexts than 131,072 were not tested with this profile (the 262,144-token
  record in the README is for the single-turn launches).
* **Very long cold prompts.** Reading one fresh 64k-100k prompt is not faster than stock vLLM's default (measured on the
  earlier multi-turn build before the FlashAttention/int8 prefill kernels: 100k tokens, 79.7 s to first token against
  46.1 s); this was not re-measured for the final bundle. The gains come from re-used history and faster decode.
* **Synthetic traces.** The speedups apply to traces like these; workloads with fewer repeated prefixes gain less.
* **Pinned FlashInfer plan buffers.** An earlier build could crash with a CUDA illegal-memory access in the
  small-chunk FlashInfer path (a race in the host plan buffer); fixed in this release by a ring of pinned buffers
  (65-minute soak with mixed caps, cancellations and cache resets: 1,873 requests, 0 failures).
* **Sampling.** As for the rest of the release, only greedy decoding is qualified.
* **JIT warm-up.** The first start compiles the CUDA kernels; `atdev_serve.warmup` then compiles the Triton kernels of the
  tree / prefix-cache path before your first request (about 4.7 s saved with an empty Triton cache). Set `NO_WARMUP=1` to skip.
