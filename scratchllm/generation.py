"""Text generation utilities for sampling from a GPT model.

Two sampling strategies are provided:

* ``generate_text_simple`` uses pure greedy decoding (argmax). It is the
  smallest possible sampler and is useful for early experiments and tests
  where determinism matters.
* ``generate`` adds temperature scaling and top-k filtering, plus an
  optional early-exit when an end-of-sequence id is produced.

Both functions assume the model returns logits of shape
``(batch, seq_len, vocab_size)`` and take token ids as a tensor of shape
``(batch, seq_len)``.
"""

import torch


def text_to_token_ids(text, tokenizer):
    """Encode a string into a ``(1, seq_len)`` int64 tensor of token ids."""
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    """Decode a ``(1, seq_len)`` or ``(seq_len,)`` token id tensor to a string."""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Greedy autoregressive generation.

    At each step, runs the model on the most recent ``context_size`` tokens,
    picks the argmax over the final-position logits, and appends it.

    Args:
        model: any module mapping ``(batch, seq_len)`` to logits.
        idx: starting token ids, shape ``(batch, seq_len)``.
        max_new_tokens: number of tokens to append.
        context_size: max context window the model can consume.

    Returns:
        Token ids of shape ``(batch, seq_len + max_new_tokens)``.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None,
):
    """Autoregressive generation with temperature and top-k sampling.

    Args:
        model: any module mapping ``(batch, seq_len)`` to logits.
        idx: starting token ids, shape ``(batch, seq_len)``.
        max_new_tokens: number of tokens to append.
        context_size: max context window the model can consume.
        temperature: if > 0, scales logits before softmax to control sharpness.
                     ``0.0`` falls back to greedy argmax (deterministic).
        top_k: if set, keeps only the top-k logits per step and masks the rest
               to ``-inf`` before sampling.
        eos_id: if set, generation stops as soon as this token is produced.

    Returns:
        Token ids of shape ``(batch, seq_len + n)`` where n <= max_new_tokens.
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
