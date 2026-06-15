# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A personal ML/AI cookbook. I work through books, tutorials, and my own
ideas here and keep the parts worth reusing. It serves three goals, in
priority order:

1. **Cookbook.** A durable, organized record of what I have learned.
   Recipes should stay runnable and self-contained.
2. **Portfolio.** This is public and meant to show I have studied and
   understand these topics. The writing matters as much as the code (see
   Voice rules).
3. **Apple Silicon first.** Target MPS, then CUDA, then CPU.

Teach as we go. The goal is for me to understand the technology, not just
to get something working.

## Sources

The repo is organized one top-level directory per source material, with
`scratchllm/` as the shared reusable library across them.

| Directory | Source | Status |
| --- | --- | --- |
| `notebooks/` | Sebastian Raschka, *Build a Large Language Model (from Scratch)* | complete (01-06) |
| `domain_specific/` | Guglielmo Iozzia, *Domain-Specific Small Language Models* | in progress |

To add a new source: create a new top-level directory for it and add a
row here. Do not rename the existing directories.

## Voice rules

The prose is part of the portfolio, so it has to read as if I wrote it.

* **No em dashes.** Use commas, parentheses, or two sentences.
* First person where natural. Plain and direct, no marketing or LLM
  filler ("it's worth noting", "delve", "leverage" as a verb).
* Comments explain why, not what. Assume the reader knows Python.
* Define a concept in a sentence before using it.
* Match the tone of the existing `notebooks/` cells, which set the style.

## Environment

Single `uv`-managed venv on Python 3.13. All dependencies live in
`pyproject.toml` (flat list); `uv.lock` is committed.

```bash
uv sync        # creates .venv (3.13), installs everything, scratchllm editable
```

In PyCharm, set the project interpreter to `.venv`. Notebooks run through
that interpreter, so after changing dependencies run `uv sync` then
Restart Kernel to pick the change up. `requirements.txt` and the old
`env-llm` venv are gone; do not reintroduce them.

## Devices

Pick a device as MPS, then CUDA, then CPU rather than hardcoding one. Some
ops are missing on MPS (set `PYTORCH_ENABLE_MPS_FALLBACK=1` when needed)
and float64 is unsupported. Note any MPS workaround in a comment so the
recipe stays reproducible.

## Layout

```
scratchllm/        Shared library (source of truth for the from-scratch code)
  model.py         LayerNorm, GELU, FeedForward, MultiHeadAttention,
                   TransformerBlock, GPTModel
  data.py          GPTDatasetV1, create_dataloader_v1
  generation.py    generate_text_simple, generate, text_to_token_ids,
                   token_ids_to_text
  training.py      train_model_simple, evaluate_model, calc_loss_batch,
                   calc_classification_loss_batch, calc_loss_loader,
                   generate_and_print_sample
  weights.py       assign, load_weights_into_gpt
  plotting.py      plot_losses
notebooks/         Raschka book, 01-06 in build order
domain_specific/   Iozzia book: fine_tuning/, inference/, onnx/
data/              the-verdict.txt and instruction-tuning JSON
gpt_download.py    GPT-2 weight downloader (Apache 2.0, third-party)
lm-evaluation-harness/   Vendored EleutherAI eval harness (third-party)
```

Notebooks build a concept by hand for teaching, then import the
consolidated version from `scratchllm`. Edit the package, do not duplicate
shared code back into notebooks.

## Artifacts

Keep large or regenerable things out of git: `*.pth` checkpoints, `gpt2/`,
Hugging Face output dirs (`*-finetuned/`, `checkpoint-*/`, `results/`,
`saved/`), `media/`, `onnx_models/`, `*.onnx`, and the SMS spam dataset
and csv splits. If you notice a heavy artifact already tracked by git,
flag it rather than adding more.
