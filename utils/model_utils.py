import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.utils import make_grid


def get_device():
    """Return the appropriate device (GPU if available, otherwise CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, path='model'):
    """Save the model's state dictionary to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}.")


def save_image(x, file_name):
    x = x.clamp(-1, 1).add(1).div(2).mul(255).byte()
    x = make_grid(x)
    x = Image.fromarray(x.permute(1, 2, 0).cpu().numpy())
    x.save(f'Image {file_name}.png')


def plot_and_save_losses(training_losses, validation_losses=None, save_path="loss_plot"):
    plt.figure(figsize=(10, 6))

    epochs = range(0, len(training_losses) + 1, 10)
    plt.xticks(epochs)

    plt.plot(training_losses, label="Training Loss", color='blue')
    if validation_losses:
        plt.plot(validation_losses, label="Validation Loss", color='orange')

    plt.title(f"Training and Validation Loss Over {len(training_losses)} Epochs", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}.png")
