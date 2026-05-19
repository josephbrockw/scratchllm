"""Loss visualization helper."""

import matplotlib.pyplot as plt


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """Plot train and validation loss against epochs, with tokens on a twin x-axis.

    Saves to ``loss-plot.pdf`` and also displays the figure inline. The
    secondary x-axis is invisible (alpha=0); it exists only to draw the
    "Tokens seen" labels above the plot at the matching positions.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig("loss-plot.pdf")
    plt.show()
