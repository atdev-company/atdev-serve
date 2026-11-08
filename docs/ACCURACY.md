# Accuracy by released profile

All broad scores below are actual saved greedy generations, with thinking disabled. Percentages are observed scores, not statistical equivalence or a universal accuracy guarantee. HumanEval/MBPP include actual EvalPlus execution. No experimental recovery checkpoint is shipped.

| Metric | Stock INT4 reference | C1 fast: original W3 | C1 accuracy: W3 disabled | C32 Hadamard W4A4 |
|---|---:|---:|---:|---:|
| MMLU-Pro (1,400) | 71.71% | 69.14% | 71.43% | 72.79% |
| KMMLU Korean (1,125) | 66.84% | 63.91% | 66.22% | 65.07% |
| IFEval prompt strict (541) | 82.90% | 82.07% | 82.99% | 82.44% |
| IFEval prompt loose (541) | 86.60% | 84.84% | 86.32% | 86.51% |
| IFEval instruction strict (834) | 87.89% | 87.65% | 88.01% | 88.01% |
| IFEval instruction loose (834) | 90.89% | 89.69% | 90.77% | 91.01% |
| HumanEval (164) | 95.73% | 94.51% | 95.73% | 95.12% |
| HumanEval+ (164) | 90.85% | 88.41% | 91.46% | 89.63% |
| MBPP (378) | 91.27% | 89.68% | 92.06% | 92.06% |
| MBPP+ (378) | 76.46% | 75.66% | 76.98% | 78.04% |
| WikiText-2 PPL, 40,920 tokens (lower is better) | 7.298721 | 8.219982 | 7.300041 | 7.209287 |
| GSM8K-300 | — | 290/300 (96.67%) | 291/300 (97.00%) | — |
| GSM8K-1319 | 1248/1319 (94.62%) | 1253/1319 (95.00%) | 1250/1319 (94.77%) | 1249/1319 (94.69%) |
| Long-context retrieval | — | 36/36 | 36/36 | 36/36 |

## Multi-turn profiles

`serve_c1.sh` (default, `multiturn`) and `serve_c32.sh` (default, `multiturn`) add prefix caching and chunked-prefill kernels to the
C1 fast and C32 Hadamard stacks. Their gate and broad scores were measured on the frozen bundles with the prefix cache on and are listed in
[MULTITURN.md](MULTITURN.md#accuracy-of-the-multi-turn-profiles). In short: C1 `multiturn` matches C1 fast within sample noise
(GSM8K-1319 94.24% against stock INT4 94.31%, long-context 36/36, MMLU-Pro 69.0%, KMMLU 63.8%, perplexity 8.28). C32 `multiturn`
(same rotated sidecar, rotation-aware 4-bit prefill GEMM) gives GSM8K-1319 94.39%, long-context 36/36, MMLU-Pro 71.4%, KMMLU 67.6%,
HumanEval+ 89.0%, perplexity 7.365; its IFEval prompt-strict score is 81.4% (mean of 3 runs: 80.2 / 82.3 / 81.7), 2.0 points under stock INT4 (83.4%); run-to-run spread is up to about 4 points.

## Speed and comparison limits

| Released profile | Speed | vs best stock-vLLM historical speed | vs original bf16 historical speed |
|---|---:|---:|---:|
| C1 fast, original published headline | 408.9 mean decode tok/s per request | 2.08x (196.5) | up to 14.2x (28.9 rounded baseline) |
| C1 accuracy, two measured repetitions | 357.172870 mean decode tok/s per request | 1.82x (196.5) | 12.4x (28.9) |
| C32 Hadamard W4A4, one measured run | 2688.128833 aggregate output tok/s | 1.53x (1757.1) | 4.0x (671.8) |

- These broad benchmark values are stored measurements of the corresponding profiles. Release packaging additionally receives startup/API checks; those checks are not a fresh full-benchmark rerun.
- C1 speed uses c=1, C32 speed uses c=32 and a different metric; do not compare their numbers as per-request speeds. Baseline and candidate figures are not all from a single session. The C32 Hadamard speed is a single run, not a repeated mean.
- Stock broad accuracy is the mean of two saved INT4-reference runs. Its GSM8K reference is a separate stock-vLLM run at c=32; C1 candidates were evaluated at c=1. Missing cells mean no corresponding measured result, not zero.
- C1 fast retains a PPL and broad-accuracy gap to stock INT4. Its maximum-speed claim does not apply to C1 accuracy. C1 accuracy still uses quantized verification and is not bit-identical to stock INT4.
- The selected C32 Korean score is the later repeated result, 65.07%; an initial run was 66.04%. Neither is a losslessness guarantee. The final selected score is about 1.78 percentage points below the two-run stock INT4 mean.
- The development GSM8K/long-context gates were narrow. Passing those gates does not mean all broader recovery gates pass; selection of these release profiles is an explicit speed/accuracy tradeoff.

## Data and integrity

C1 W3 and drafter weights are unchanged from the original release. C1 accuracy does not require a new target checkpoint. C32 adds only the selected Hadamard GPTQ sidecar, SHA256 `608aef5bf2569e183845cd61ecb783385f61b491603912002d753edcca14728b`. Calibration and training provenance is in [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md).

Evidence identifiers: original `EVAL-broad/c1`; W3-disabled `EVAL-broad/accnow3b` (final audit and two-run speed); selected C32 `ACC-c32/gen/had3_hadw_gptq_d10` and `P1b-fqts4k3di1rcalbcpsz-acchad3_hadw_gptq_d10`. These are experiment identifiers, not paths required on an installation machine.
