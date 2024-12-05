import torch

from model_classes.model_conditional_batch_norm import ModelConditionalBatchNorm
from utils.data_loaders import load_dataset_and_make_dataloaders
from utils.pipeline_utils import get_device, build_sigma_schedule, compute_c_functions, sampling_pipeline
from utils.workflow_utils import animate_denoising, save_grid_image

device = get_device()
model = ModelConditionalBatchNorm(image_channels=1, nb_channels=128, num_blocks=6, cond_channels=32)
model.load_state_dict(torch.load('../model_classes/models/MCBN_ch128_b8_cond64.pth'))
model.to(device)

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
model.to(device)
print(model)

save_grid_image(image, 'loaded_model_images')
animate_denoising([img.cpu() for img in intermediate_images], save_path="loaded_model_denoising")
