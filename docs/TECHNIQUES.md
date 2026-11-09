# Techniques, measured effects and failure cases

Every technique that ships in the four profiles (**c1m**, **c1s**, **c32m**, **c32s**, see the [README](../README.md#the-four-profiles)),
what it does, what we measured, and what went wrong when we tried it. Gains are throughput changes measured in the development
tree on one A100 80GB (vLLM 0.31.0, greedy), usually with an A/B in the same session; they are not additive, and a few were measured
on an earlier state of the stack. "Exact" means bit-identical output to the path it replaces. Techniques that change numerics are
called out; the accuracy results for the profiles are in [ACCURACY.md](ACCURACY.md) and [MULTITURN.md](MULTITURN.md).

How the pieces fit together is in [HOW_IT_WORKS.md](HOW_IT_WORKS.md); this page is the lab notebook behind it.

Contents: [shared by c1m and c1s](#c1-shared-by-c1m-and-c1s) - [c1m only](#c1m-only-multi-turn-c1) - [shared by c32m and c32s](#c32-shared-by-c32m-and-c32s) -
[c32m only](#c32m-only-multi-turn-c32) - [optional: end-of-reply checkpoint](#optional-end-of-reply-checkpoint) - [tried, did not ship](#tried-and-not-shipped) -
[operational lessons](#operational-lessons)

---

## c1: shared by c1m and c1s

| Technique | What it does | Measured effect | Exact? |
|---|---|---|---|
| Token-tree verification (`dflash_tree`, `tree_attn`, `gdn_tree_cuda`) | DFlash2 drafts a block of 16; the target verifies a 16-node tree in one forward pass (ancestor-masked attention, tree recurrence for the GDN layers) | the core of the c=1 gain; 4.69 accepted tokens per step | yes: acceptance is exact greedy |
| Re-trained block-16 drafter, W4A16 (`drafter_w4`) | self-distilled on target-generated answers; int4 weights | block 16 instead of 8: +3.8%; int4 drafter +3.3% | drafter only proposes |
| Single-state GDN verify (`gdn_single_state`) | one recurrent state per step plus a token log; accepted tokens are replayed | removes the per-draft-position state copies | yes |
| 64k draft vocabulary (`draft_vocab`, `shadow_head`) | candidate scoring on the 64k most frequent tokens with an int4 copy of those lm_head rows | +7.5% | yes (the target verifies every token) |
| Split verify head (`verify_split_head`) | verify lm_head = int8 rows for the 64k frequent tokens + int4 rows for the rest | +4.4%, then +3.5% | no (model-side numerics) |
| W3 overlay (`w3_mixed`, fused W3 Marlin) | 400 linear layers in 3 bits | MLP layers +6.5%, remaining layers +4.3%, kernel v2 and tuning +3.4% | no (model-side) |
| Fused epilogues (`c1_fuse`), `marlin_plus` launch configs, `ba_gemv`, split-KV verify attention | norm / residual / SiLU folded into GEMM epilogues; tuned tiles for M=16 | +3.5% (c1_fuse), +2.4% (marlin_plus) | yes |

Cautions and failure cases from this family:

* **Wider trees.** A 24, 32 or 48 node tree accepts more tokens (+7.5% at 32 nodes) but the 32-row verify GEMM costs about 20% more than the
  16-row one on the A100, so the net effect was 0.73x-0.90x. 16 nodes stays.
* **A deeper drafter block.** Block 32 (depth 31) was measured offline: nothing past depth 15 is reachable with a 16-node tree, and the
  accepted length fell 5% on the block-16 weights. Block 16 is the sweet spot.
* **Drafter layer drop.** Removing one drafter layer saves about 2.3% of the step but lost 7% of the accepted length: killed.
* **Exact certified screen for the verify head.** A certified int4 screen with exact rescoring needs one host synchronization per verify
  step, which breaks asynchronous scheduling (+2.3 ms per step): 0.86x. Killed. The shipped split head is approximate by design and is gated on accuracy.
* **n-gram / prompt-lookup candidates.** Merging n-gram children into the tree gave +1.0 to +1.5% offline accepted length at best, and
  replacing tree nodes by a chain-swap lost 12% when the match was short. Not shipped.
* **Markov head on tree edges.** Memorized the bigrams of the training text (-13.6% accepted length without regularization, about 0% with it).
* **Overlap tricks at M=16** (PDL emulation, L2 prefetch, two-stream GEMMs): at most about 2 microseconds per GEMM boundary, since one M=16 GEMM already fills the GPU.
* **W4A4 at c=1** (M=16): neutral speed with a smaller accuracy margin.

## c1m only (multi-turn c1)

| Technique | What it does | Measured effect | Exact? |
|---|---|---|---|
| Prefix caching for the tree stack (`ATDEV_DFT_APC`, `gdn_ss_align`) | the recurrent state is re-materialized at aligned positions from the single-state slot plus the token log; the tree verifier re-bases its state tables | prerequisite for everything below; hit rate 0.95 on the 111k-token trace (stock vLLM with prefix caching reaches the same hit rate) | not byte-identical to cache-off (prefill split at block edges) |
| Fine-grained hits for the drafter's sliding-window layers (`ATDEV_SWA_FINE`) | 16-token granularity instead of block granularity | part of the 0.95 hit rate | yes |
| FlashAttention-2 prefill (`ATDEV_PF_FA2`) | prefill and mixed steps through FA2 instead of the Triton unified kernel | the Triton kernel ran prefill at 10-13% of A100 peak; FA2 is 3.1-3.8x faster on this shape | rounding-level difference |
| FlashInfer split-KV for short tails (`ATDEV_PF_FA2_FI`, `..._GATHER`) | the 2..299-token steps at 100k context under-fill the FA2 tiles; FlashInfer's balanced split-KV needs 0.3-4 ms where FA2 needs 4.6-9 ms per layer | about 1.2% of the trace for the small-q path | rounding-level |
| W3 int8 prefill (`ATDEV_W3_I8`, `ATDEV_W3_DQ`) | large chunks: the W3 weights are dequantized once to int8 and run on the int8 tensor cores | part of the c1m bundle; not isolated as a separate number | no (int8 activations) |
| Prefill glue (`ATDEV_PF_FUSE`, `ATDEV_PF_GLUE`) | fused SiLU-mul / residual-RMSNorm with the int8 quantization | removes about 380 ms of 1.56 s for a 3.8k-token prefill | yes for PF_FUSE, tested end to end for GLUE |
| Small-chunk GDN kernel (`ATDEV_GDN_SMALL`) | recurrent kernel for GDN chunks of 32 tokens or less | the 13-token tails of a turn no longer pay the chunked kernel | rounding-level |
| Tree attention head split for long context (`ATDEV_TREE_HSPL=auto`) | one CTA per KV head: 100k context 16 layers 11.7 ms to 5.3 ms | decode at long context | not bit-identical to the Triton kernel |
| Tokenization cache (`ATDEV_TOK_CACHE`) | per-segment tokenization of the growing history | removes the repeated tokenization of the growing history; not isolated as a separate number | yes |
| Start-up warmup (`atdev_serve.warmup`) | synthetic conversations compile the Triton kernels before the first request | a 3-turn probe trace 9.5 s to 4.8 s with an empty Triton cache (12 kernels were JIT-compiled during the first request, 4.7 s) | yes |

Failure cases and cautions:

* **FlashInfer `plan()` pinned-buffer race (fixed).** `plan()` stages its work list in one pinned host buffer and copies it to the device
  asynchronously. With the CPU running ahead (async scheduling), a heavy 2048-token chunk followed by two tiny steps let the second `plan()` overwrite
  the buffer before the first copy ran, and the first step's kernels read the wrong work list: a CUDA illegal memory access eight seconds after a cache reset.
  Fix: a ring of 8 pinned buffers with a CUDA event per slot (`ATDEV_PF_FA2_FI_PINRING=1`, default). Outputs are unchanged (all 76 turns identical, speed unchanged);
  a 65-minute soak (1,873 requests, concurrency 1-4, mixed caps, 182 cache resets, 201 aborted requests) showed no failure. The same check was done for the C32 FlashInfer path.
* **int8 KV cache (removed).** A split-KV int8 kernel ran correctly but the cache returned corrupt output after a prefix-cache hit (stock vLLM's int8 KV showed the same). It was
  also not faster than the retuned bf16 kernel at c=1, and capacity does not bind on these traces. The code was removed from this release; the profiles keep a bf16 KV cache.
* **MBT 4096.** A 4096-token prefill chunk crashed in combination with the (then unfixed) FlashInfer race; 2048 is the shipped value. Not retried after the fix.
* **Sage-style int8 QK attention.** Needs a custom kernel; estimated gain about 8 s of 205 s, not built.
* **Memory.** The dequantized int8 copy of the W3 weights is why c1m holds about 284k KV tokens at 131,072 context where c1s holds about 540k.
* **Very long cold prompts.** A single fresh 64k-100k prompt is not faster than stock vLLM's default (measured on an earlier multi-turn build; not re-measured on the final one).

## c32: shared by c32m and c32s

| Technique | What it does | Measured effect | Exact? |
|---|---|---|---|
| Confidence-scheduled variable-length MTP verify (`di1_varlen`) | MTP drafts up to 3 tokens but the verify batch keeps the row budget of k=2 (96 rows at 32 requests) | tokens per step 2.24-2.31 to 2.41-2.47: +2.1 to +2.9% | yes |
| W4A4 verify GEMMs (MarlinA4) with fused quantizing producers (`fused_quant`, `gdn_a4_fold`) | 4-bit weights and 4-bit activations on int4 tensor cores; norm / SiLU / gate kernels write the 4-bit activations directly | the large GEMMs are about 54% of the c=32 step | no (activation quantization) |
| Hadamard rotation of the weights and activations (c32s and c32m) | block-Hadamard rotated GPTQ weights remove the activation outliers that W4A4 cannot represent | accuracy class of stock INT4 (see ACCURACY) | model-side |
| Variable-length GDN verify kernel (`gdn_ss12h`, `gdn_ss15h`) | one state per request, longest-processing-time CTA ordering | first version +5.75% | yes |
| CUDA varlen verify attention (`c32_attn`), conv update (`c32_conv`) | replace the Triton kernels, bit-identical | +1.7% | yes |
| W4A4 screen head (`vsplit_a4scr`) | int4 screen over all 248k lm_head rows picks 16 candidates; those are rescored against the original bf16 rows | head time 1.34 ms to 0.63 ms at M=96: +2.2% | candidates only |
| int4 MTP head, pruned first pass, argmax-only verify (`mtp_w4`, `mtp_prune`, `vsplit_mtp`) | | +1.8% | yes |
| 64k int4 draft vocabulary | | +3% | yes |

Failure cases and cautions:

* **Fixed MTP k=3.** Verifying every draft costs +20% step time for +15.9% tokens per step: the confidence scheduler keeps the k=2 budget instead.
* **Unrotated W4A4.** The earlier C32 release used W4A4 on unrotated weights; its accuracy gap on MMLU-Pro and Korean knowledge is why the Hadamard sidecar exists.
* **A4 GEMM tuning that stayed flat.** The W4A4 prefill GEMM runs at 36% of int4 peak (about 53% of the measurable peak once the clock drops to 1.13 GHz). Time ablations show the FP32 fold work is hidden
  behind the tensor-core work. Tried without any effect beyond noise: L2 tile ordering, delaying the second CTA of each SM, interleaving the fold behind the next MMA (best -1.1%), a 2-K-group pipeline stage (slower: +6 to +14% on the large shapes), a wider config sweep (29 instantiations),
  requant refinements (-1.9% to -2.5% wall for the exact-reciprocal quantizer, the only part that stuck). We could not read hardware counters on the test machine, so the stall analysis is based on clock instrumentation and microbenchmarks.
* **A CUTLASS int4 GEMM** reaches about 1000 TOPS without the per-group fold, but the C1 W3 weights bake group scales into int8 values and cannot use a per-channel int4 GEMM.

## c32m only (multi-turn C32)

| Technique | What it does | Measured effect | Exact? |
|---|---|---|---|
| Prefix caching for MTP (`ATDEV_GDN_SS_ALIGN`) with the shared-prefix checkpoint | single-state GDN verify under `--mamba-cache-mode align` | hit rate 0.76 on the 32-session trace | not byte-identical to cache-off |
| FlashAttention-2 prefill (`ATDEV_PF_FA2`) plus tuned Marlin configs for M > 128 (`ATDEV_PF_MP`) | | 1.16x / 1.26x over stock on the 32 x 6 trace at caps 128 / 512, before the W4A4 prefill GEMM | rounding-level / exact |
| FlashInfer paged attention for the prefill rows of mixed steps (`ATDEV_PF_FI`) | one host plan per step; decode rows stay on the CUDA varlen kernel through a decode-only metadata view | -5.6% / -3.8% wall at caps 128 / 512 | rounding-level (max abs diff 0.125 vs FA2) |
| W4A4 prefill GEMM (`ATDEV_PF_A4`) on rotated weights, fused rotation, direct s4 emission, clip 0.9, layer 0 on 8-bit activations | prefill chunks use the int4 tensor cores | on the unrotated stack it saved 11-20% of the v5-c32 wall time against the int8 prefill path; on the rotated stack the total is within 0.5% of the unrotated bundle's speed | no (activation quantization on rotated weights) |
| Tokenization cache (`ATDEV_TOK_CACHE`) | | neutral on the C32 trace | yes |
| Start-up warmup with 32-way waves | compiles the batch-shape kernels before the first request | removed a +5% first-run penalty | yes |

Failure cases and cautions:

* **FlashInfer on decode rows (wrong output).** A first version planned the whole batch with FlashInfer. Decode rows came out wrong, because the confidence scheduler decides
  the number of draft rows per request on the GPU (the GPU's query start offsets differ from the CPU copy) while FlashInfer fixes its tile layout at plan time. A GSM8K probe inside the trace collapsed to roughly 0.1-0.3. The shipped path plans only the prefill-sized rows and leaves the decode rows on the stock path.
* **W4A4 prefill without rotation.** The first W4A4 prefill GEMM on stock INT4 weights raised WikiText-2 perplexity from 7.30 to 7.96 and lost 2-4 points on MMLU-Pro, KMMLU and HumanEval+. The loss was spread over every layer and projection class (keeping any single class on 8 bit still left 7.5-7.8),
  so selective 8-bit layers on those weights did not work.
* **Double rotation (fixed).** On the rotated weights the first W4A4 prefill attempt gave perplexity 2e7 (and 3e7 for one projection): the Hadamard wrapper left its "rotate" flag set for the GDN out-projection at large batch, so the quantizer rotated an already-rotated activation a second time. The flag is now
  set only on the small-batch path. After the fix the perplexity is 7.37 (8-bit prefill on the same weights: 7.21).
* **Rotation overhead.** The unfused rotation made the rotated stack 17-23% slower than the unrotated bundle. A fused rotate-and-quantize kernel (SiLU-mul in 0.41 ms instead of 2.92 ms at M=4096) brought it back to +0.5%.
* **Things that did not help accuracy:** emitting the 4-bit fragments directly (the double rounding is not the loss); recalibrating the rotated GPTQ weights on knowledge and Korean text (no gain, wiki perplexity +0.03); keeping the decode down-projection on 8 bit (no gain beyond noise, costs decode speed); per-layer 8-bit lists (the per-layer ranking is below the measurement floor; only layer 0 stands out, so layer 0 is kept on 8 bit).
  An activation clip of 0.9 recovers 12-22% of the gap between 4-bit and 8-bit activations at no speed cost.
* **Noise.** One configuration's Korean-knowledge score varies by about 1.4 points run to run (n=1,125), and its IFEval prompt-strict score by up to about 4 points; claims below about 1.5 points need several runs or a larger proxy.
* **Batch limits.** `--max-num-batched-tokens` 3072 was 1.5% / 1% slower than 4096, and a verify row budget of 128 instead of 96 gave no gain (+0.3% / +1% wall). 4096 and 96 are shipped.

## Optional: end-of-reply checkpoint

`ATDEV_EOR_CKPT=1` (off by default; c1m and c32m). vLLM caches state only up to the end of the prompt, so the next turn of a conversation recomputes the server's own previous reply. The checkpoint snapshots the recurrent state at 16- or 32-token boundaries while decoding and registers the newest one under the hash of the generated tokens.

* **Benefit.** C32 live-reply trace (each arm feeds back its own replies): -5.2% / -12.3% wall at caps 128 / 512 with spacing 16, -2.8% / -7.9% with spacing 32. C1: neutral at cap 128, about -7% at cap 512 (indicative).
* **Cost.** On traces where replies cannot be reused: +1.3% / +2.7% with spacing 32, and about 3% more GPU memory (`--gpu-memory-utilization` 0.93).
* **Cache-page eviction.** The first version cost +8% / +50% (caps 128 / 512) on the recorded-reply trace. The kernel time was small; the cause was the cache: 32 lock-step sessions sit at the LRU cliff of the pool, and about 100 extra registered pages per finished turn evicted the checkpoints the next turn needed (hit rate 0.80 to 0.59-0.65).
  Fixes that worked: snapshot spacing 32 instead of 16, a smaller ring, a planned two-stage kernel, and the extra 3% of memory. Fixes that did not work reliably: bounding the registered pages FIFO-style.
* **Hit granularity and exactness.** The cached state ends a few tens of tokens before the reply's end; continuing from it is not byte-identical to recomputing (in correctness checks 4 of 10 C32 and 5 of 7 C1 cases matched token for token, mean log-probability difference 0.01-0.10).

## Tried and not shipped

* **C1 3-bit recovery.** The W3 overlay costs 2.5-3.3 points on MMLU-Pro and Korean knowledge against stock INT4. Attempts to recover accuracy without losing speed did not pass our gates:
  scaling each block output to match the INT4 model (WikiText-2 perplexity 8.16 against 8.22 for W3 and 7.29 for INT4: modest, rejected); restoring 56 of the 64 layers to 4 bits and keeping 8 layers in 3 bits (MMLU-Pro recovered to 71.2, Korean knowledge still -1.5, and 359 tok/s, 12% slower than c1s: rejected);
  a refit of scales and codes against the INT4 teacher (perplexity 7.89, speed 408.3 against 411.7 for the matched baseline: not enough accuracy, rejected); an asymmetric 3-bit continuation calibrated layer by layer against the INT4 model (perplexity 8.56, worse than W3: rejected); a fixed block-Hadamard rotation before 3-bit quantization, the method that works for c32 activations (quantization error of the down projection 0.158 to 0.150 in an offline check: no useful gain, because the C32 fix targets 4-bit activation error while the C1 loss comes from 3-bit weights). A whole-chain diagnostic showed why layer-local fits fall short: each layer's error is small but it accumulates over 64 layers (calibration NLL 4.59 for the INT4 teacher against 4.87 for the 3-bit student). 2-bit MLP layers on top of W3 lost 2-4 points of GSM8K for 0.6-1.0 GB saved.
  Further recalibration work is in progress; no result yet.
* **No-W3 c1m (measured, not shipped).** With the stock INT4 target (no W3 and no int8 prefill paths) the multi-turn C1 stack ran the 111k trace at cap 512 in 250.3 s (2.49x faster than the fastest stock setting at 623.1 s; 1.22x slower than c1m), with a 519k-token KV cache. Speed only; no accuracy run. It is not a released profile.
* **int8 KV, sparse attention, wider trees, deeper blocks:** see above.
* **CPU-side draft tricks (n-gram, Markov head):** see above.

## Operational lessons

* **Stale torch-extension build locks.** A server killed during a JIT build leaves a `lock` file in its extension directory; every later start of a process using that directory then waits on it forever, with the GPU idle (about 70 W) and no log output after the build line.
  Remove the lock when no `ninja` / `nvcc` is running (`TORCH_EXTENSIONS_DIR/*/lock`), and keep one extension directory per tree.
* **Draft-vocabulary / verify-head defaults.** The release bakes in a 64k int4 draft vocabulary and a 64k exact + int4 tail verify head. The research code reads these from environment variables whose defaults differ (16-bit and 32k / 8-bit). Running the wrong defaults changed the generated text of 20 of 76 turns
  of the 111k trace (same speed, same GSM8K), which is how the mismatch was found: **output identity against a frozen run is a stricter check than speed or GSM8K**. The launch scripts export the shipped values explicitly.
* **`ATDEV_VSPLIT_FAST` with the c1 side stream.** The argmax-only verify path allocates a stock Marlin workspace lazily on a side stream, which trips vLLM's "workspace is locked" assertion on the first request. The c1 profiles must not set it; the c32 profiles use their own tail path.
* **First-run JIT.** A fresh Triton cache costs about +5% on the first C32 run and 4.7 s on the first C1 request; the warmup step is part of the multi-turn launchers for this reason.
* **Typos and unread flags.** `ATDEV_OPT` with an unknown name stops the server, so a typo cannot silently fall back to stock vLLM; flags that nothing reads are checked in the repository's tests.
