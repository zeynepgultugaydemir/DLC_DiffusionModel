import difflib
import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
from matplotlib import animation
from torchvision.utils import make_grid
from torchvision.utils import save_image


def get_device():
    """Return the appropriate device (GPU if available, otherwise CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, path='model'):
    """Save the model's state dictionary to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}.")


def save_generated_images(images, folder, prefix):
    transform = T.Normalize(mean=[-1], std=[2])
    for i, img in enumerate(images):
        img = transform(img).clamp(0, 1)
        save_image(img, os.path.join(folder, f"{prefix}_{i}.png"))


def save_grid_image(x, file_name):
    x = x.clamp(-1, 1).add(1).div(2).mul(255).byte()
    x = make_grid(x)
    x = Image.fromarray(x.permute(1, 2, 0).cpu().numpy())
    x.save(f'{file_name}.png')


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
    fig, axes = plt.subplots(batch_size, num_levels, figsize=(num_levels * 1.5, batch_size * 1.5))

    for i in range(batch_size):
        for j in range(num_levels):
            image = noisy_images[j, i].squeeze().cpu().numpy()

            ax = axes[i, j] if batch_size > 1 else axes[j]
            ax.imshow(image, cmap="gray" if channels == 1 else None)
            ax.axis("off")

            if i == 0:
                ax.set_title(f"σ={sigmas[j]:.2f}", fontsize=12, pad=5)

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig('sample_noising.png', bbox_inches="tight")
    plt.show()


def animate_denoising(intermediate_images, save_path='denoising', interval=50):
    fig, ax = plt.subplots(figsize=(5, 5))

    def update(i):
        ax.clear()
        image = intermediate_images[i][0, 0].numpy()
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        ax.set_title(f"Step {i + 1}")

    ani = animation.FuncAnimation(fig, update, frames=len(intermediate_images), interval=interval)

    if save_path:
        ani.save(f'{save_path}.gif', writer="pillow")
    plt.show()


def map_name_to_class(class_name):
    class_synonyms = {
        0: ["t-shirt", "t-shirt", "tshirt", "top", "tee"],
        1: ["trouser", "trousers", "pants", ''],
        2: ["pullover", "sweater", "jumper"],
        3: ["dress"],
        4: ["coat", "jacket"],
        5: ["sandal", "sandals", "flip-flop", "flip-flops", "high heels"],
        6: ["shirt", "button-up", "blouse"],
        7: ["sneaker", "sneakers", "trainer", "trainers", "running shoe", "running shoes"],
        8: ["bag", "handbag", "purse"],
        9: ["ankle boot", "ankle boots", "boot", "boots"],
    }

    class_mapping = {synonym.lower(): class_number for class_number, synonyms in class_synonyms.items() for synonym in synonyms}
    class_name = class_name.lower()

    if class_name in class_mapping:
        print(f'Generating images of {class_name}...')
        return class_mapping[class_name]

    possible_matches = list(class_mapping.keys())
    closest_match = difflib.get_close_matches(class_name, possible_matches, n=1, cutoff=0.6)

    if closest_match:
        return f"Class name {class_name} not found. Did you mean '{closest_match[0]}'?"
    else:
        return "Class name not found."
