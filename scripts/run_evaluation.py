import torch
from pytorch_fid import fid_score

from model_classes.model_conditional_batch_norm import ModelConditionalBatchNorm
from utils.data_loaders import load_dataset_and_make_dataloaders
from utils.pipeline_utils import get_device, build_sigma_schedule, compute_c_functions, sampling_pipeline
from utils.workflow_utils import save_generated_images

if __name__ == '__main__':

    device = get_device()

    model = ModelConditionalBatchNorm(image_channels=1, nb_channels=128, num_blocks=8, cond_channels=64)
    model.load_state_dict(torch.load('../model_classes/models/MCBN_img1_ch128_b8_cond64.pth', map_location=torch.device('cpu')))
    model.to(device)

    dl, info = load_dataset_and_make_dataloaders(dataset_name='FashionMNIST', root_dir='../data', batch_size=128,
                                                 num_workers=0, pin_memory=device)
    real_folder = "../images/fid_evaluation/real_images"
    generated_folder = "../images/fid_evaluation/generated_images"

    for y, label in dl.train:
        images = y
        save_generated_images(y, real_folder, "real")
        print(y.shape)
        print(label.shape)
        break

    sigmas = build_sigma_schedule(50)
    c_funcs = compute_c_functions(sigmas, info.sigma_data)

    image, intermediate_images = sampling_pipeline(model, images, sigmas, info.sigma_data, device=device)
    model.to(device)
    save_generated_images(image, generated_folder, "generated")

    fid = fid_score.calculate_fid_given_paths([real_folder, generated_folder], batch_size=128, device=device, dims=2048)
    print(f"FID score: {fid:.2f}")
