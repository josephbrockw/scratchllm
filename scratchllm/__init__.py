"""scratchllm: a GPT-style decoder-only transformer built from scratch in PyTorch.

This package is the consolidated home for every reusable building block
that the notebooks demonstrate. The notebooks build each piece up step by
step for pedagogy; this package is what they import once those pieces have
been introduced.

Public surface (also accessible via the submodules):

* Model:     ``LayerNorm``, ``GELU``, ``FeedForward``, ``MultiHeadAttention``,
             ``TransformerBlock``, ``GPTModel``
* Data:      ``GPTDatasetV1``, ``create_dataloader_v1``
* Generation:``generate_text_simple``, ``generate``,
             ``text_to_token_ids``, ``token_ids_to_text``
* Training:  ``calc_loss_batch``, ``calc_classification_loss_batch``,
             ``calc_loss_loader``, ``evaluate_model``, ``train_model_simple``,
             ``generate_and_print_sample``
* Weights:   ``assign``, ``load_weights_into_gpt``
* Plotting:  ``plot_losses``
"""

from .data import GPTDatasetV1, create_dataloader_v1
from .generation import (
    generate,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)
from .model import (
    FeedForward,
    GELU,
    GPTModel,
    LayerNorm,
    MultiHeadAttention,
    TransformerBlock,
)
from .plotting import plot_losses
from .training import (
    calc_classification_loss_batch,
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    generate_and_print_sample,
    train_model_simple,
)
from .weights import assign, load_weights_into_gpt

__all__ = [
    "FeedForward",
    "GELU",
    "GPTDatasetV1",
    "GPTModel",
    "LayerNorm",
    "MultiHeadAttention",
    "TransformerBlock",
    "assign",
    "calc_classification_loss_batch",
    "calc_loss_batch",
    "calc_loss_loader",
    "create_dataloader_v1",
    "evaluate_model",
    "generate",
    "generate_and_print_sample",
    "generate_text_simple",
    "load_weights_into_gpt",
    "plot_losses",
    "text_to_token_ids",
    "token_ids_to_text",
    "train_model_simple",
]
