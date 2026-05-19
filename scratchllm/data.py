"""Dataset and dataloader for next-token prediction.

The training task is to predict each next token in a sequence given all
previous tokens. We tokenize a text corpus once with the GPT-2 BPE
tokenizer, then slide a fixed-size window across the token stream, taking
each window as an input and the same window shifted by one as the target.
"""

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    """Sliding-window dataset over BPE-tokenized text.

    Each example is a pair of tensors of shape ``(max_length,)`` representing
    one input window and the same window shifted right by one token.

    Args:
        txt: raw text to tokenize.
        tokenizer: a tokenizer with ``.encode(str) -> list[int]``.
        max_length: window length in tokens.
        stride: how many tokens to advance the window between examples.
                Use ``stride < max_length`` for overlapping windows (more
                examples, more redundancy), or ``stride == max_length`` for
                non-overlapping windows.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    """Build a PyTorch DataLoader over a sliding-window view of ``txt``.

    Uses the GPT-2 BPE tokenizer via tiktoken. Returns batches of
    ``(input_ids, target_ids)``, each shape ``(batch_size, max_length)``.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
