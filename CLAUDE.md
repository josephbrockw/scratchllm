# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview

Educational project implementing a GPT-style decoder-only transformer
from scratch in PyTorch, following Sebastian Raschka's _Build a Large
Language Model (from Scratch)_. Organized as a small library
(`scratchllm/`) plus six guided notebooks (`notebooks/`).

**Important**: please explain everything and teach as we work through
issues. The goal is for me to understand the technology more than to get
something working.

## Environment

```bash
python -m venv env-llm
source env-llm/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `from scratchllm import ...` work
```

Python 3.13. Virtual environment lives in `env-llm/` (gitignored).

## Repo layout

```
scratchllm/        Python package, the consolidated reusable code
  __init__.py      re-exports the public surface
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
notebooks/         01 through 06, in build order
data/              the-verdict.txt and the instruction-tuning JSON files
gpt_download.py    GPT-2 weight downloader (Apache 2.0, third-party)
```

The notebooks build concepts up from scratch and then import the
consolidated version from `scratchllm` once a piece has been
introduced. When making changes, the package is the source of truth;
notebooks should import from it rather than duplicate it.

## Notebook progression

1. **`01_preprocessing.ipynb`**: regex tokenizer with a hand-built vocab,
   then BPE via `tiktoken`. Sliding-window dataset and dataloader. Token
   and position embeddings.
2. **`02_attention.ipynb`**: simplified dot-product attention, then
   self-attention with learned Q/K/V, then causal masking, then
   multi-head attention.
3. **`03_architecture.ipynb`**: LayerNorm, GELU, FeedForward, residual
   connections, TransformerBlock, full GPTModel.
4. **`04_pretraining.ipynb`**: training loop, train/val split, periodic
   eval, temperature and top-k sampling, loading OpenAI GPT-2 weights.
5. **`05_classification.ipynb`**: fine-tune GPT-2 for binary spam
   classification (freeze body, replace head, train on last-token
   logits).
6. **`06_instruction.ipynb`**: supervised instruction fine-tuning on
   1,100 Alpaca-style examples, with padding-aware collation.

## Data and checkpoints

Data files committed to the repo (in `data/`):

* `the-verdict.txt`: short story used as the pretraining corpus.
* `instruction-data.json`: 1,100 instruction-tuning examples.
* `instruction-data-with-response.json`: 110 held-out examples with
  model responses.

Local-only, gitignored:

* `gpt2/` directory of OpenAI GPT-2 TF checkpoints (regenerate via
  `gpt_download.download_and_load_gpt2(...)`).
* `*.pth` checkpoint files (regenerate by running the training cells in
  notebooks 04, 05, 06).
* Downloaded SMS spam dataset under `data/sms_spam_collection*`
  (notebook 05 redownloads on first run).

## Key dependencies

* **torch**: model and training
* **tiktoken**: GPT-2 BPE tokenizer
* **numpy**: used by the GPT-2 weight loader
* **tensorflow**: only needed by `gpt_download.py` to read the
  published GPT-2 TF checkpoints
* **matplotlib**: loss plots
* **pandas**: only used in notebook 05 for the SMS spam dataset
