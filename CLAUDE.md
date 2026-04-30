# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational project implementing LLM components from scratch in PyTorch, following the "LLMs from scratch" curriculum. Work is organized as Jupyter notebooks that build up concepts incrementally.

*IMPORTANT* please make sure to explain everything and teach as we work through issues. The goal is for me to understand the technology more so than getting something to work.

## Environment Setup

```bash
python -m venv env-llm
source env-llm/bin/activate
pip install -r requirements.txt
```

Python virtual environment lives in `env-llm/` (already in `.gitignore`). Uses Python 3.13.

## Key Dependencies

- **torch** (2.11.0) — all tensor operations and neural network layers
- **tiktoken** (0.12.0) — BPE tokenizer (GPT-2 encoding)
- **numpy**, **jupyter**, **ipykernel**

## Notebook Progression

1. **preprocessing.ipynb** — Tokenization (custom vocab + BPE via tiktoken), sliding-window data loading (`GPTDatasetV1`/`DataLoader`), token + positional embeddings
2. **attention.ipynb** — Simplified dot-product attention, softmax normalization, self-attention with trainable Q/K/V weight matrices, scaled attention

## Training Data

`the-verdict.txt` — short story used as the corpus for all tokenization and data loading examples.
