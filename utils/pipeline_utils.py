import datetime

import numpy as np
import torch

from utils.data_loaders import get_data_folder_path, load_dataset_and_make_dataloaders
from utils.workflow_utils import get_device


def sample_sigma(n, loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=80):
    return (torch.randn(n) * scale + loc).exp().clip(sigma_min, sigma_max)


def build_sigma_schedule(steps, rho=7, sigma_min=2e-3, sigma_max=80):
    """
    Build a schedule of decreasing noise levels (sigmas)

    Parameters
    ----------
    steps: Number of steps in the schedule.
    rho: Controls the rate of decay of the noise levels
    sigma_min: Minimum noise level
    sigma_max: Maximum noise level

    Returns
    ----------
    Sequence of sigmas (noise levels) decreasing progressively
    """
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def add_noise(clean_image, sigma):
    """
    Add Gaussian noise to a clean image based on the specified noise level (sigma).

    Parameters
    ----------
    clean_image: Tensor of clean images
    sigma: Noise levels for each image in the batch

    Returns
    ----------
        Noisy images after adding Gaussian noise.
    """
    # Expand sigma to match the shape of clean_image for element-wise operations
    sigma_expanded = sigma.view(-1, 1, 1, 1).expand_as(clean_image)
    return clean_image + sigma_expanded * torch.randn_like(clean_image)  # Gaussian noise ~ N(0, 1)


def denoise(model, x, c_funcs, class_vector=None):
    """
    Denoise an input image using the model and provided scaling functions.

    Parameters
    ----------
    model: model to be trained
    x: Noisy input images
    c_funcs: Scaling functions of c_in, c_out, c_skip and c_noise
    class_vector: Class conditioning vector if model is class conditioned

    Returns
    ----------
        Denoised output images
    """
    c_in, c_out, c_skip, c_noise = c_funcs

    # Forward pass through the model
    if class_vector is not None:
        try:
            x_denoised = model.forward(c_in * x, c_noise, class_vector=class_vector)
        except TypeError:
            x_denoised = model.forward(c_in * x, c_noise)
    else:
        x_denoised = model.forward(c_in * x, c_noise)

    # Combine the model's output with the skip connection
    return c_skip * x + c_out * x_denoised


def compute_c_functions(sigma, sigma_data):
    """
    Compute the scaling functions based on the noise level provided

    Parameters
    ----------
    sigma: Noise levels
    sigma_data: Standard deviation of the dataset

    Returns
    ----------
        tuple of scaling functions
    """
    c_in = 1 / torch.sqrt(sigma ** 2 + sigma_data ** 2)
    c_out = (sigma * sigma_data) / torch.sqrt(sigma ** 2 + sigma_data ** 2)
    c_skip = (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)
    c_noise = torch.log(sigma) / 4

    return c_in, c_out, c_skip, c_noise


def training_pipeline(model, dataset_name="FashionMNIST", batch_size=32, epochs=5, learning_rate=0.001,
                      validation=False, device=None, demo=False, num_classes=None):
    """
        Train a denoising model on the specified dataset.

    Parameters
    ----------
    model: Model to be trained
    dataset_name: Name of the dataset to use
    batch_size: Batch size for training/validation
    epochs: Number of training epochs
    learning_rate: Learning rate for the optimizer
    validation: If True, run with validation data
    device: Device to run the training (CPU/GPU).
    demo: If True, limit training to one batch for demonstration purposes
    num_classes: Number of classes for if the model is class conditioned

    Returns
    ----------
        Mean training/validation losses for each epoch.
    """
    device = device or get_device()

    data_path = get_data_folder_path()
    dl, data_info = load_dataset_and_make_dataloaders(dataset_name=dataset_name, root_dir=data_path,
                                                      batch_size=batch_size, drop_last=True)

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    data = dl.valid if validation else dl.train
    all_loss = []
    for epoch in range(epochs):
        model.eval() if validation else model.train()
        training_loss = []
        print(datetime.datetime.now())
        for y, labels in data:
            # Sample noise levels and compute scaling functions
            sigma = sample_sigma(y.shape[0]).to(device)
            c_in, c_out, c_skip, c_noise = compute_c_functions(sigma, data_info.sigma_data)
            c_in = c_in.view(-1, 1, 1, 1).expand(-1, y.size(1), y.size(2), y.size(3))
            c_out = c_out.view(-1, 1, 1, 1).expand(-1, y.size(1), y.size(2), y.size(3))
            c_skip = c_skip.view(-1, 1, 1, 1).expand(-1, y.size(1), y.size(2), y.size(3))

            y = y.to(device)
            x = add_noise(y, sigma).to(device)

            # Prepare class vector for conditional models
            if num_classes is not None:
                labels = labels.to(device)
                class_vector = torch.zeros((labels.size(0), num_classes), device=device)
                class_vector.scatter_(1, labels.view(-1, 1), 1)
            else:
                class_vector = None

            # Forward pass
            if class_vector is not None:
                output = model.forward(c_in * x, c_noise, class_vector=class_vector)
            else:
                output = model.forward(c_in * x, c_noise)

            target = (y - c_skip * x) / c_out
            loss = criterion(output, target)

            # Backpropagation and optimization (only if training)
            if not validation:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            training_loss.append(loss.item())

            if demo:  # Break early for demonstration purposes
                break

        all_loss.append(np.mean(training_loss))  # collecting the mean error for each epoch
        print(f"Epoch {epoch + 1}, {'Validation' if validation else 'Training'} Loss: {np.mean(training_loss):.4f}")

    return all_loss


def sampling_pipeline(model, images, sigmas, sigma_data, device, target=None, conditional=None):
    """
        Sampling pipeline to generate images using a trained model


    Parameters
    ----------
    model: Trained denoising model
    images: Input images to pass the shape
    sigmas: Sequence of noise levels for sampling
    sigma_data: Standard deviation of the dataset
    device: Device to run the pipeline (CPU/GPU).
    target: Target class labels if model is class conditioned
    conditional: If True, enable class conditioning

    Returns
    ----------
        Final denoised images and intermediate results for visualisations
    """
    intermediate_images = []
    num_classes = 10
    batch_size = images.size(0)

    # Start with pure Gaussian noise
    x = (torch.randn(*images.shape, device=device) * sigmas[0])

    if conditional:
        # Prepare class vector for conditional generation
        if target:
            class_vector = torch.zeros((batch_size, num_classes), device=device)
            class_vector[torch.arange(batch_size), target] = 1
        else:
            # Conditional model without target class input, randomly generating a mix of classes
            class_vector = torch.randint(0, 2, (batch_size, num_classes), device=device, dtype=torch.float32)
            class_vector /= class_vector.sum(dim=1, keepdim=True) + 1e-6
    else:
        class_vector = None

    # Iteratively denoise using the model
    for i, sigma in enumerate(sigmas):
        with torch.no_grad():
            c_in, c_out, c_skip, c_noise = compute_c_functions(sigma, sigma_data)

            x_denoised = denoise(model, x, c_funcs=(c_in.to(device).view(1, 1, 1, 1), c_out.to(device).view(1, 1, 1, 1),
                                                    c_skip.to(device).view(1, 1, 1, 1), c_noise.to(device).view(1)),
                                 class_vector=class_vector)

        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d = (x - x_denoised) / sigma
        x = x + d * (sigma_next - sigma)
        intermediate_images.append(x_denoised.detach().cpu())

    return x, intermediate_images
