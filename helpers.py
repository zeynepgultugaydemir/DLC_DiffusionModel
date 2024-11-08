import torch

from data_loaders import load_dataset_and_make_dataloaders, get_data_folder_path


def sample_sigma(n, loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=80):
    return (torch.randn(n) * scale + loc).exp().clip(sigma_min, sigma_max)


def build_sigma_schedule(steps, rho=7, sigma_min=2e-3, sigma_max=80):
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def build_noisy_image(clean_image, sigma):
    noise = sigma * torch.randn_like(clean_image)
    return clean_image + noise, noise


def compute_c_functions(sigma, sigma_data):
    c_in = 1 / torch.sqrt(sigma ** 2 + sigma_data ** 2)
    c_out = (sigma * sigma_data) / torch.sqrt(sigma ** 2 + sigma_data ** 2)
    c_skip = (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)
    c_noise = torch.log(sigma) / 4

    return c_in, c_out, c_skip, c_noise


def get_device():
    """Return the appropriate device (GPU if available, otherwise CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, path):
    """Save the model's state dictionary to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}.")


def training_pipeline(model, dataset_name="FashionMNIST", batch_size=32, epochs=5, learning_rate=0.001, device=None):
    """
        Train a denoising model on the specified dataset.
    """
    # TODO finish up the pipeline & comments

    device = device or get_device()
    data_path = get_data_folder_path()
    dl, data_info = load_dataset_and_make_dataloaders(dataset_name=dataset_name, root_dir=data_path,
                                                      batch_size=batch_size)

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    for epoch in range(epochs):
        for y, _ in dl.train:
            sigma = sample_sigma(y.shape[0]).to(device)

            y = y.to(device)
            x, noise_level = build_noisy_image(y, sigma)
            x = x.to(device)

            c_in, c_out, c_skip, c_noise = compute_c_functions

            output = model.forward(c_in * x, c_noise)
            target = (y - c_skip * x) / c_out

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Noise Level: {noise_level:.2f}, Loss: {loss.item():.4f}")
