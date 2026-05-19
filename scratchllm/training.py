"""Training loop, loss functions, and evaluation utilities.

Two loss shapes are supported:

* ``calc_loss_batch`` for next-token prediction (language modeling). Cross
  entropy is taken over every position in the sequence.
* ``calc_classification_loss_batch`` for sequence classification. Only the
  logits at the final position are used, treating the model as a feature
  extractor whose last hidden state encodes the whole sequence.

``calc_loss_loader`` accepts whichever per-batch loss function via the
``loss_fn`` argument so the evaluation loop is shared.
"""

import torch

from .generation import generate_text_simple, text_to_token_ids, token_ids_to_text


def calc_loss_batch(input_batch, target_batch, model, device):
    """Cross-entropy loss over every position for language modeling.

    Flattens batch and sequence dims so cross_entropy sees a 2D input.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )


def calc_classification_loss_batch(input_batch, target_batch, model, device):
    """Cross-entropy on the final-position logits for sequence classification.

    Uses only ``logits[:, -1, :]`` because in a causal model the last
    position is the only one that has attended over the entire sequence.
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_loss_loader(data_loader, model, device, num_batches=None, loss_fn=None):
    """Average a per-batch loss across (a prefix of) a dataloader.

    Args:
        loss_fn: per-batch loss function. Defaults to ``calc_loss_batch`` for
                 language modeling. Pass ``calc_classification_loss_batch``
                 for classification.
        num_batches: limit evaluation to this many batches for fast estimates.
    """
    if loss_fn is None:
        loss_fn = calc_loss_batch
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = loss_fn(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model, train_loader, val_loader, device, eval_iter, loss_fn=None
):
    """Compute train and validation loss without leaking eval-mode noise.

    Switches to eval mode (disabling dropout), evaluates, then restores
    train mode so the caller can keep training.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter, loss_fn=loss_fn
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter, loss_fn=loss_fn
        )
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate a short sample from ``start_context`` and print it.

    Used as a training-loop hook so we can watch the model's outputs evolve
    qualitatively across epochs.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    """Vanilla training loop with periodic evaluation and sample generation.

    Returns three lists captured at each evaluation step: train losses,
    validation losses, and the cumulative number of tokens seen at that point.
    These are convenient to plot.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen
