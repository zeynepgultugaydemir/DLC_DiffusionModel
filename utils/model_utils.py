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


def plot_noising(noisy_images, sigmas):
    noisy_images = noisy_images.flip(0)
    sigmas = sigmas.flip(0)

    num_levels, batch_size, channels, height, width = noisy_images.shape

    fig, axes = plt.subplots(batch_size, num_levels)

    for i in range(batch_size):
        for j in range(num_levels):
            image = noisy_images[j, i].squeeze().cpu().numpy()

            ax = axes[i, j] if batch_size > 1 else axes[j]
            ax.imshow(image, cmap="gray" if channels == 1 else None)
            ax.axis("off")

            if i == 0:
                ax.set_title(f"σ={sigmas[j]:.2f}", fontsize=10)

    plt.show()
