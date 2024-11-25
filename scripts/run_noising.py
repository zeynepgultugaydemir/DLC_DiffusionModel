import torch

from utils.data_loaders import load_dataset_and_make_dataloaders
from utils.pipeline_utils import build_sigma_schedule, get_device, add_noise
from utils.workflow_utils import plot_noising

device = get_device()

dl, info = load_dataset_and_make_dataloaders(dataset_name='FashionMNIST', root_dir='../data', batch_size=8, num_workers=0, pin_memory=device)

for y, label in dl.train:
    images = y.to(device)
    print("Image batch shape:", y.shape)
    print("Label batch shape:", label.shape)
    break

sigmas = build_sigma_schedule(50, sigma_max=50).to(device)
indices = torch.linspace(0, len(sigmas) - 1, steps=10).long()

noisy_images = [add_noise(images, sigma) for sigma in sigmas]
noisy_images_tensor = torch.stack(noisy_images, dim=0)

plot_noising(noisy_images_tensor[indices], sigmas[indices])
