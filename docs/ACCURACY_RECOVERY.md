# Accuracy recovery validation — 2026-11-01

The two development profiles below recover the primary MMLU-Pro and KMMLU scores to within
one percentage point of the two-run stock INT4 mean on one A100 80GB. This is evidence in the
measured scope, not proof of statistical equivalence, bit-identical answers, or universal losslessness.

**Release status:** this is a documentation and aggregate-results update. The existing
`serve_c1.sh` / `serve_c32.sh` launchers and uploaded Hugging Face weights still describe the
legacy configurations. They do **not** reproduce the profiles below without the corresponding
runtime/configuration changes. In particular, the c32 rotated sidecar is not uploaded in this
update. Do not combine the new accuracy scores with the legacy 408.9 / 2,730.9 tok/s claims.

## Actual measured configurations

- **c1 / now3b:** original RedHatAI INT4 target body; all additional target W3 paths disabled.
  The clean d16 W4A16/fc8 drafter, clean2 token order, F8 frequent / int4 tail verify head,
  and tree/engine optimizations are retained. The verify head remains approximate.
- **c32 / Had3 + down W4A8:** block-Hadamard128 rotated GPTQ INT4 weights (dampening 0.1,
  group size 128, sourced from AWQ-smoothed bf16 weights), Had3 W4A4 activations with only
  `down_proj` activations restored to W4A8; built-in MTP k=3, confidence-scheduled verification,
  clean2 token order. Its 12,543,701,944-byte rotated sidecar has SHA256
  `608aef5bf2569e183845cd61ecb783385f61b491603912002d753edcca14728b`.

Broad evaluation uses the actual served concurrency (1 or 32), temperature 0 and thinking disabled.
The same configuration was checked across accuracy and speed stages. Each task has its complete
expected unique IDs, no final request failures, and nonmissing token counts. HumanEval and MBPP
include actual EvalPlus execution, not just generated code. GSM scores were independently
recomputed from saved answers. No dataset questions or generated benchmark answers are included
in this public evidence update.

## Accuracy

| metric | stock INT4 run 1 / run 2 | c1 now3b | c32 Had3 + down W4A8 |
|---|---:|---:|---:|
| MMLU-Pro (1,400) | 71.50% / 71.93% | 71.43% | 71.71% |
| KMMLU (1,125) | 67.11% / 66.58% | 66.22% | 67.20% |
| IFEval prompt strict (541) | 83.36% / 82.44% | 82.99% | 84.47% |
| IFEval prompt loose (541) | 87.06% / 86.14% | 86.32% | 87.99% |
| HumanEval base (164) | 95.12% / 96.34% | 95.73% | 95.73% |
| HumanEval+ (164) | 90.24% / 91.46% | 91.46% | 92.68% |
| MBPP base (378) | 91.53% / 91.01% | 92.06% | 89.68% |
| MBPP+ (378) | 76.72% / 76.19% | 76.98% | 76.72% |
| WikiText-2 perplexity (40,920 tokens) | 7.29872 / 7.29872 | 7.30004 | 7.20929 |
| GSM8K all 1,319 | 94.62% (one stock run) | 94.77% | 94.54% |
| Long-context retrieval | — | 36/36 | 36/36 |

The primary floors were fixed before the final measurements: MMLU-Pro >=70.7143%,
KMMLU >=65.8444% (two-run stock mean minus 1 percentage point). PPL must be <=1.02x
stock; full GSM8K must be within 1 point of the actual stock 94.6171% reference; longctx 36/36.
C1 GSM8K first 300 is separately reported: 97.00%, versus its legacy 97.33% reference.
Secondary scores and paired comparisons are disclosed, rather than inferred from GSM alone.

C32 primary knowledge was generated twice: MMLU-Pro 71.6429% / 71.7143%, KMMLU
67.7333% / 67.2000%. C1 full broad generation was measured once. C32 MBPP base is lower
than the two stock runs by 1.85 / 1.32 points (paired approximate 95% CIs
[-4.2,+0.5] / [-3.7,+1.1] points). MBPP+ is near stock. A CI including zero does not prove
no degradation. All point differences and paired CIs are in the comparison tables in this folder.
These benchmarks were used to select configurations; this is a development evaluation, not a
blind held-out claim. No evaluation examples were intentionally included in the listed
training/calibration pools.

### Scoring conditions

IFEval uses a deterministic adapter: language-detector seed 0, and literal counting of the
explicitly requested `#` / `!` characters instead of upstream random-letter fallback.
Both stock reference runs and candidates were scored with the same adapter. These numbers
must be distinguished from unmodified upstream IFEval scoring. The metadata is
`ifeval_langdetect_seed=0`, `ifeval_literal_punctuation=true`. Two CPU regression tests pass;
the existing seven complete answer sets were rescored twice with identical per-item outcomes.
The scorer change received independent code review.

## Speed and tradeoff

96 fixed prompts (39 Korean), up to 1,024 output tokens, warmup before each repetition.
C1 reports mean per-request decode tok/s; c32 reports aggregate output tok/s.

| | c1 | c32 |
|---|---:|---:|
| repetition 1 | 357.1408 | 2,325.5975 |
| repetition 2 | 357.2049 | 2,319.6511 |
| mean | **357.1729** | **2,322.6243** |
| vs stored best vLLM (196.5 / 1,757.1) | **1.818x** | **1.322x** |
| loss vs legacy public stack (408.9 / 2,730.9) | **12.65%** | **14.95%** |

The denominators are historical stored results, not fresh same-session baseline measurements.
Accuracy recovery required giving up some legacy throughput. W3 recalibration and mixed
precision were investigated; mix56 passed proxies but failed real KMMLU (65.33%) and was
rejected. The c32 down+out W4A8 branch was pruned for lower measured speed.

## Evidence and provenance

- [Machine-readable aggregate results](accuracy_recovery.json): counts, audit checks, all metrics,
  repeated speed values and fixed primary thresholds.
- [Data and license inventory](DATA_AND_LICENSES.md): training versus calibration versus evaluation.
- Paired comparisons: [c1 vs run 1](c1_vs_refint4.md), [c1 vs run 2](c1_vs_refint4b.md),
  [c32 vs run 1](c32_vs_refint4.md), [c32 vs run 2](c32_vs_refint4b.md).
