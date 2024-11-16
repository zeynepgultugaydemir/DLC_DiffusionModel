import torch

from model_classes.model import Model
from utils.data_loaders import load_dataset_and_make_dataloaders
from utils.pipeline_utils import get_device, build_sigma_schedule, \
    compute_c_functions, sampling_pipeline
from utils.workflow_utils import animate_denoising, save_image

device = get_device()
model = Model(image_channels=1, nb_channels=64, num_blocks=3, cond_channels=32).to(device)
model.load_state_dict(torch.load('../model_classes/models/model.pth'))

dl, info = load_dataset_and_make_dataloaders(dataset_name='FashionMNIST', root_dir='../data', batch_size=32,
                                             num_workers=0, pin_memory=device)

for y, label in dl.train:
    images = y
    print(y.shape)
    print(label.shape)
    break

sigmas = build_sigma_schedule(50, sigma_max=50)
c_funcs = compute_c_functions(sigmas, info.sigma_data)

image, intermediate_images = sampling_pipeline(model, images, sigmas, info.sigma_data, device=device)

# save_image(image, 'sampling')
animate_denoising([img.cpu() for img in intermediate_images], save_path="denoising")
