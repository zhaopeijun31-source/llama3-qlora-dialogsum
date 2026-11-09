# QLoRA Fine-tuning of Llama-3-8B-Instruct for Dialogue Summarization

Fine-tuned **Meta-Llama-3-8B-Instruct** on a dialogue summarization task with **QLoRA**
(4-bit NF4 quantization + LoRA), using a single free **Kaggle T4 GPU (15 GB)**.

## Results

Evaluated on the first 100 samples of the test set (greedy decoding).

| Metric  | Base model | Fine-tuned | Change |
|---------|-----------:|-----------:|-------:|
| ROUGE-1 | 0.2097 | **0.4846** | +0.27 (×2.3) |
| ROUGE-2 | 0.0692 | **0.2093** | +0.14 (×3.0) |
| ROUGE-L | 0.1662 | **0.3990** | +0.23 (×2.4) |

Validation loss decreased steadily (no overfitting):

| Step | Training loss | Validation loss |
|-----:|--------------:|----------------:|
| 50  | 1.197 | 1.438 |
| 100 | 1.106 | 1.375 |
| 150 | 1.152 | 1.351 |
| 200 | 1.127 | 1.341 |
| 250 | 1.180 | 1.337 |

### Example

| | Summary |
|---|---|
| **Reference** | #Person1# attends Brian's birthday party. Brian thinks #Person1# looks great and charming. |
| **Base model** | The conversation is about a birthday celebration. Person1 is giving a gift to Person2, who is celebrating their birthday. Person1 asks to dance with Person2 and they both seem to be enjoying the party. Person2 compliments Person1 on their appearance and they both agree that the party is wonderful. Person1 suggests having a drink together to celebrate the birthday. |
| **Fine-tuned** | #Person1# gives Brian a birthday gift and they dance together. #Person1# thinks Brian is popular and looks great. |

The fine-tuned model follows the reference style (short, third-person, `#Person1#` placeholders) and stops on its own.
It still makes factual errors, e.g. swapping who compliments whom, which ROUGE does not capture.

In another sample, the base model even repeats the instruction back as a list of questions and runs into the token limit, while the fine-tuned model answers in two sentences:

| | Summary |
|---|---|
| **Reference** | #Person1# thinks #Person2# has chicken pox and warns #Person2# about the possible hazards but #Person2# thinks it will be fine. |
| **Base model** | Please summarize the conversation between #Person1# and #Person2#. What is the main topic of the conversation? What is the concern of #Person1#? ... *(continues until the 150-token limit)* |
| **Fine-tuned** | #Person1# thinks #Person2# has chicken pox and is contagious. #Person2# thinks it's just a rash or an allergy. |

All 100 predictions are in [`results/predictions.csv`](results/predictions.csv).

## Setup

| Item | Value |
|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Dataset | [`neil-code/dialogsum-test`](https://huggingface.co/datasets/neil-code/dialogsum-test) (DialogSum subset: 1,999 train / 499 validation / 499 test) |
| Quantization | 4-bit NF4, double quantization, fp16 compute |
| LoRA | r=64, alpha=16, dropout=0.01, target: `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable parameters | 54.5M (~0.7% of 8B) — adapter file: 209 MB |
| Training | 1 epoch, effective batch size 8 (1 × 8 gradient accumulation), lr 2e-4, paged AdamW 8-bit, gradient checkpointing |
| Hardware / time | 1 × NVIDIA T4 (15 GB), ~65 min of training |

Data analysis: dialogues average ~190 tokens and summaries ~33 tokens (max sample ~1,200 tokens), so `max_length=1024` was used.

## Fixes to the reference implementation

This project started from a tutorial written for Llama-2 on a 24 GB GPU. Running it with Llama-3 on a T4 exposed several issues:

| Problem | Cause | Fix |
|---|---|---|
| Fine-tuned model never stops generating | `add_eos_token=True` is ignored by Llama-3's tokenizer, so training texts had no EOS | Append `<|eot_id|>` to each training text explicitly |
| EOS would be excluded from the loss | `pad_token` was set to the EOS token; the collator masks pad tokens | Use `<|end_of_text|>` as a separate pad token |
| Train/inference prompt mismatch | Different prompt templates were used for training and testing | One `make_prompt()` function shared by both |
| OOM during training (tried to allocate 1.96 GiB) | `prepare_model_for_kbit_training` casts embedding and `lm_head` to fp32; with Llama-3's 128k vocabulary each is 128256 × 4096 × 4 B = 1.96 GiB | Skip it; enable gradient checkpointing and input gradients manually |
| OOM at evaluation | `per_device_eval_batch_size` defaults to 8 | Set to 1 |
| CUBLAS error at evaluation | Trainer wrapped the 4-bit model in `DataParallel` on the 2 Kaggle GPUs | `CUDA_VISIBLE_DEVICES=0` |
| Deprecated arguments | Newer `transformers` renamed `evaluation_strategy` | Pinned library versions, updated arguments |
| `CUDA VERSION MISMATCH` in bitsandbytes | Kaggle's image moved to PyTorch with CUDA 13.0; bitsandbytes 0.47 has no CUDA 13 build | Use `bitsandbytes>=0.48.0` |

## How to run

1. Request access to Llama-3 on Hugging Face and create a read token.
2. In Kaggle: import `qlora_llama3_dialogsum.ipynb`, set Accelerator = **GPU T4 x2**, Internet = **On**, and add the token as a Secret named `HF_TOKEN`.
3. Run all cells (or *Save & Run All* to run in the background, ~2 hours in total).

Outputs: the LoRA adapter (`llama3-dialogsum-qlora-adapter/`) and `results/` (ROUGE scores, predictions, training log).

## Limitations and future work

- The loss is computed on the whole text (prompt + dialogue + summary); computing it only on the summary may improve results.
- Evaluated on 100 test samples only, with ROUGE, which does not measure factual correctness.
- Hyperparameters (e.g. `r`) were not tuned.

## Repository structure

```
├── qlora_llama3_dialogsum.ipynb   # full pipeline: data → 4-bit model → LoRA → training → evaluation
├── requirements.txt
└── results/
    ├── rouge_scores.json
    ├── predictions.csv            # reference / base / fine-tuned summaries for 100 test samples
    └── training_log.csv
```
