# Training, calibration and licenses

The released choices are the original C1 W3 fast profile, the existing W3-disabled C1 accuracy profile, and the selected C32 Hadamard W4A4 profile. Research-only recovery candidates are not included.

## Models and code

- The serving plugin is Apache-2.0; the existing `LICENSE`, `NOTICE` and third-party notices remain in force.
- Target sources: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4), Apache-2.0.
- C1 drafter source: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), Apache-2.0. The released drafter is the existing self-distilled, quantized d16 checkpoint; it has not been retrained for this update.
- Model derivatives are distributed under Apache-2.0 with these upstream attributions. Dataset-source attribution is recorded separately below; this does not relicense source datasets as Apache-2.0.

## C1 fast and C1 accuracy

The original W3 overlay and DFlash2 drafter files are unchanged. C1 accuracy uses the stock RedHatAI INT4 target and the same existing drafter; disabling W3 does not create a new trained target checkpoint.

The C1 prompt pool uses [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) `train_sft` (MIT) and [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) `main/train` (MIT), plus author-written agent/answer-language wrappers. Target-generated responses were used for self-distillation and quantization calibration. Original overlap filtering against evaluation prompts, GSM8K test and MATH-500 is documented on the [C1 model card](https://huggingface.co/atdev-company/Qwen3.8-27B-atdev-c1).

The original C1 W3 calibration used 400 prompt/response pairs: 200 UltraChat, 105 UltraChat with Korean instructions, 46 UltraChat with agent instructions, 43 GSM8K train, and 6 GSM8K train with Korean instructions. These were truncated to 1,024 tokens (348,936 tokens total), with 8,192 sampled activation rows per linear layer. The separate C32 recipe below must not be confused with this C1 recipe.

## C32 Hadamard W4A4

This is offline quantization/calibration, not an additional language-model fine-tuning run. The selected artifact records `bits=4`, `group=128`, source `bf16`, method `gptq_act d0.1`, and rotation `hadamard128-blockdiag-K`. It was quantized from the BF16 source in the AWQ-smoothed space required by the INT4 target. It replaces 400 projections (800 packed-weight/scale tensors); embeddings, norms, heads and other parameters remain supplied by the INT4 base. C32 uses that base model's built-in MTP and adds no external drafter checkpoint.

The calibration capture submitted 1,024 rows, with prefix length at most 1,024 tokens (780,107 tokens):

| Source group | Submitted rows | License / attribution |
|---|---:|---|
| Target-generated clean prompt/response pool (`gen_clean` 384 + `gen_clean2` 128) | 512 | UltraChat/GSM8K train prompt sources as above; author-written wrappers and target-generated responses |
| FineWeb-2 Korean (`kor_Hang`) | 256 | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), ODC-By 1.0; follow its upstream source/attribution terms |
| FineWeb-Edu (`sample/10BT`) | 256 | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), ODC-By 1.0; follow its upstream source/attribution terms |

The capture used seed 0 and a cap of 16,384 activation rows per linear layer. All 1,024 rows were submitted, but sampled row IDs were not saved and the cap can fill before the final prompt; therefore we do not assert that every submitted row contributed to every projection. The FineWeb-2 source split called `test` is a source-dataset split, not a GSM8K/MMLU/KMMLU evaluation split. These raw web sources are additional C32 calibration data and are not covered by the C1 card's “only UltraChat/GSM8K” description. Full benchmark-overlap filtering of the C32 raw web subsets was not independently established; the C1 clean-pool overlap statement must not be generalized to those web subsets.

The FineWeb dataset cards also refer to [Common Crawl terms](https://commoncrawl.org/terms-of-use). Source datasets retain their own attribution and applicable source-content terms. No raw prompt pool, raw web corpus or benchmark answer set is redistributed in this release. Benchmark prompts/answers are used for evaluation, not added to the released calibration recipe.

## Artifact integrity

- Existing C1 W3 MLP/out overlay: SHA256 `c61a3a8a976c05bdf8beddf639e2fc773ee920ff898ff9401852295802d8c8f2`.
- Existing C1 W3 QKV overlay: SHA256 `6cdcf837586fc13cdc1a70f7b7ba316f4d84d2f9f507a88f60f16231cbeb55cb`.
- Existing C1 drafter weights: SHA256 `f19850f9c7eb980badc1332c55f6741e942f19bc6e6fe2e085e792c467c52edf`.
- Added C32 Hadamard sidecar: 12,543,701,944 bytes; SHA256 `608aef5bf2569e183845cd61ecb783385f61b491603912002d753edcca14728b`.

Source calibration pool fingerprint (for provenance, not a downloadable dataset in this repo): SHA256 `2b1ad573e8587bdd19ba2efd17ce61c1ff4a6609ecba536f1530a7a760561667`.
