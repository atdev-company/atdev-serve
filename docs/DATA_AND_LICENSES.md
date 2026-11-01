# Data provenance and licensing

Checked 2026-11-01 against local scripts, generated-pool inventories, the rotated safetensors
metadata and upstream model/dataset cards. This inventory describes the artifacts actually used;
it does not relicense third-party assets or certify all possible downstream uses.

## Additional learning and generated data

The target model was not newly pretrained or instruction-fine-tuned by ATDEV for these results.
The c1 **drafter was additionally trained** by self-distillation from the target. The accuracy
recovery run retained that existing clean drafter; it did not perform another accepted drafter
training run. C32 recovery used post-training quantization/calibration and precision changes.

| source | split | role | upstream license |
|---|---|---|---|
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | `train_sft` | drafter training prompts; legacy W3 calibration; token-order statistics; part of c32 calibration | MIT |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `main/train` | same roles; **not** the test split | MIT |
| [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | `kor_Hang`, test rows >=10,000 | Korean raw text for c32 recovery quantization calibration; not drafter training | ODC-BY 1.0 |
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | `sample/10BT`, shard 000, row groups 0–2 | English raw text for c32 recovery quantization calibration; not drafter training | ODC-BY 1.0 |

The clean generation pool contains **12,000 training records**. Its prompts come only from
UltraChat and GSM8K train, with target-generated responses (greedy and sampled; some thinking
modes). Authors added short agent/system, Korean-answer and programming-language instructions.
Drafter learning starts from `z-lab/Qwen3.8-27B-DFlash2`: stage 1 block 8, learning rate 2e-4,
375 total steps, gradient accumulation 32; stage 2 block 16, learning rate 1e-4, 31 steps on
MD5 shard 0/12 of that pool. Targets use teacher hidden states / token labels. Quantization
then produces W4A16 decoder linears and int8 fc, group size 128.

The vocabulary-order statistics additionally use **4,000** target-generated UltraChat
programming-oriented records, for **16,000 total input records** across both pools. The shipped
`token_order.npy` matches the clean2 order used in validation. It contains token IDs, no source
text. The earlier exploratory A10 mixtures are not the sources of this clean release token order.

The legacy uploaded W3 overlays used 400 clean prompt/response records (348,936 tokens,
truncated to 1,024 tokens per record), 8,192 sampled activation rows per linear. They remain
available as legacy artifacts but are **disabled in the recovered c1 profile**.

The c32 rotated sidecar metadata identifies `calib_v2`, Hadamard128 rotation, GPTQ act-order,
4 bits, group 128, dampening 0.1 and bf16 source. `calib_v2` has **1,024 records**, truncated
to at most 1,024 tokens each (**780,107 tokens** in the pool):

| pool/category | records |
|---|---:|
| clean/general | 96 |
| clean/Korean | 96 |
| clean/code | 80 |
| clean/math | 56 |
| clean/agentic | 56 |
| clean2/code | 64 |
| clean2/agentic | 32 |
| clean2/Korean | 32 |
| FineWeb-2 Korean | 256 |
| FineWeb-Edu English | 256 |

This is quantization calibration, not another instruction-fine-tuning corpus. The Korean proxy
uses earlier FineWeb-2 rows (<5,000), separate from the listed calibration range.
Generation scripts applied exact/ngram filtering against their evaluation-reference lists,
including GSM8K test and MATH-500. This is a filtering procedure, not proof of zero contamination
in upstream pretrained models. Development benchmarks were also used for candidate selection.

## Evaluation-only sources

MMLU-Pro (1,400 selected test items), KMMLU (1,125 selected test items), IFEval (541),
HumanEval+ (164), MBPP+ (378), WikiText-2, GSM8K test (300 and all 1,319) and author-written
long-context retrieval checks (36) are evaluation data, not listed as additional training data.
These are selected local evaluation subsets where noted, not a claim of full MMLU-Pro/KMMLU coverage.
IFEval uses its upstream split named `train`, solely as an evaluation benchmark here; it is
not part of the drafter training pool.

- [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro): MIT upstream card.
- [KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU): **CC BY-ND 4.0**, evaluated locally;
  dataset questions, reformatted prompts and model answers are not redistributed in this update.
  The [license](https://creativecommons.org/licenses/by-nd/4.0/) includes attribution and
  restrictions on distributing adapted material; it is not MIT/Apache data.
- [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval),
  [HumanEval](https://github.com/openai/human-eval),
  [MBPP](https://github.com/google-research/google-research/tree/master/mbpp),
  [EvalPlus](https://github.com/evalplus/evalplus), and
  [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) retain their upstream terms.
  This update publishes aggregate scores and audit metadata, not benchmark source text.

## Models and code

| component | license / attribution |
|---|---|
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Apache-2.0; original target family / bf16 source |
| [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | Apache-2.0; served INT4 target |
| [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | Apache-2.0; original drafter architecture/checkpoint |
| ATDEV modifications / model derivatives | Apache-2.0; preserve upstream conditions and notices |
| vLLM / Marlin | Apache-2.0; see `NOTICE` and retained source headers |
| PyTorch-derived code | BSD-3-Clause; see `licenses/LICENSE.pytorch` in the code repository |
| flash-linear-attention-derived code | MIT; see `NOTICE` in the code repository |

Apache-2.0 for ATDEV code and derivatives does not replace the dataset licenses or other
third-party rights. Retain attribution, license texts and modification notices when redistributing.
Underlying source-model pretraining data are supplied by the upstream model authors, not newly
inventoried or independently audited by ATDEV here.

## Attribution and limitations

This work uses prompts from UltraChat / HuggingFaceH4 and GSM8K / OpenAI under their MIT
notices. See [UltraChat's dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
and [GSM8K's MIT license](https://github.com/openai/grade-school-math/blob/master/LICENSE).

Contains information from **FineWeb-2** and **FineWeb-Edu**, made available by Hugging Face
under the [Open Data Commons Attribution License (ODC-BY) v1.0](https://opendatacommons.org/licenses/by/1-0/).
FineWeb inputs were sampled, tokenized and used for calibration; no raw FineWeb corpus is
redistributed here. ODC-BY covers database rights and does not itself grant every right in each
underlying web page. Upstream source-content, privacy and other rights can still apply.

MIT license texts for the two prompt sources are preserved in the accompanying license files
(`LICENSE.ultrachat`, `LICENSE.gsm8k`); the code repository also retains PyTorch BSD and FLA MIT.

The reviewed declared licenses show no identified incompatibility with publishing this
Apache-2.0 code/model documentation and aggregate benchmark update with these notices.
This is a scoped license inventory, not an unconditional legal clearance of training data,
upstream pretraining provenance or every commercial downstream deployment.
