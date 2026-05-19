"""Model architecture for a GPT-style decoder-only transformer.

The stack composes, from bottom to top:

    LayerNorm + GELU + FeedForward   (basic building blocks)
    MultiHeadAttention                (the core sequence-mixing op)
    TransformerBlock                  (one attention + ffn layer)
    GPTModel                          (full stack with embeddings and head)

Every tensor flows as ``(batch, seq_len, emb_dim)`` between blocks. Attention
internally reshapes to ``(batch, num_heads, seq_len, head_dim)`` and projects
back. The vocabulary projection at the head produces logits of shape
``(batch, seq_len, vocab_size)``.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Per-token layer normalization with learnable scale and shift.

    Normalizes over the embedding dimension (the last axis) so each token
    has zero mean and unit variance before being affinely rescaled. Using a
    handwritten implementation rather than ``nn.LayerNorm`` is purely
    pedagogical: it makes the variance/epsilon arithmetic explicit.

    Args:
        emb_dim: size of the embedding dimension being normalized.

    Shape:
        input  ``(..., emb_dim)``
        output ``(..., emb_dim)``
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # Biased variance (unbiased=False) to match GPT-2's reference implementation.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Gaussian Error Linear Unit using the tanh approximation.

    The tanh form is what the original GPT-2 implementation uses, so we use
    it here for weight-loading compatibility. PyTorch's built-in
    ``nn.GELU(approximate="tanh")`` would behave identically.
    """

    def forward(self, x):
        return 0.5 * x * (
            1
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside each transformer block.

    Two linear layers with a 4x expansion in the hidden dimension, separated
    by GELU. Applied independently to every position in the sequence.

    Shape:
        input  ``(batch, seq_len, emb_dim)``
        output ``(batch, seq_len, emb_dim)``
    """

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention.

    Splits the embedding dimension across ``num_heads`` heads so each head
    attends over a smaller subspace. A causal mask (upper-triangular) blocks
    each position from attending to future positions, which is what makes
    this an autoregressive language model.

    Args:
        d_in: input embedding dimension.
        d_out: output embedding dimension. Must be divisible by num_heads.
        context_length: maximum sequence length the causal mask is sized for.
        dropout: dropout probability applied to attention weights.
        num_heads: number of attention heads.
        qkv_bias: whether the Q/K/V projections include a bias term.
                  GPT-2's official checkpoints use biases here, so set True
                  when loading those weights.

    Shape:
        input  ``(batch, seq_len, d_in)``
        output ``(batch, seq_len, d_out)``
    """

    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length), diagonal=1
            ),
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Split d_out across heads: (b, T, d_out) -> (b, T, H, head_dim).
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Move the head axis next to batch so attention runs per-head in parallel.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Fill future positions with -inf so softmax sends them to exactly zero.
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    """One transformer block: pre-norm attention plus pre-norm feed-forward.

    Pre-norm means LayerNorm is applied to the residual stream before the
    sublayer, not after. Residual connections wrap each sublayer. Dropout is
    applied to each sublayer output before it's added back to the stream.

    Shape:
        input  ``(batch, seq_len, emb_dim)``
        output ``(batch, seq_len, emb_dim)``
    """

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    """A small GPT-style decoder-only transformer.

    Composes token embeddings, learned positional embeddings, a stack of
    transformer blocks, a final layer norm, and an unembedding projection
    back to the vocabulary.

    The config dict must include: ``vocab_size``, ``emb_dim``,
    ``context_length``, ``drop_rate``, ``n_layers``, ``n_heads``, and
    ``qkv_bias``.

    Shape:
        input  ``(batch, seq_len)`` int64 token ids
        output ``(batch, seq_len, vocab_size)`` logits
    """

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)
