import torch
from pytorch_fid import fid_score

from utils.config import MODEL_CONFIGS
from utils.data_loaders import load_dataset_and_make_dataloaders
from utils.pipeline_utils import get_device, build_sigma_schedule, sampling_pipeline
from utils.workflow_utils import animate_denoising, save_grid_image, map_name_to_class, save_generated_images


def run_sampling(model_config, evaluate=False):
    target = None
    device = get_device()

    if model_config['conditional']:
        while True:
            user_input = input("Enter the desired target class name (e.g., 't-shirt', 'dress', 'sneaker', etc.): ").strip()

            if user_input != '':
                mapped_class = map_name_to_class(user_input)
                if isinstance(mapped_class, int):
                    target = mapped_class
                    break
                elif isinstance(mapped_class, str):
                    print(mapped_class)
            else:
                print("No input provided. Proceeding without a specified class.")
                break

    dl, info = load_dataset_and_make_dataloaders(dataset_name='FashionMNIST', root_dir='../data', batch_size=32, num_workers=0, pin_memory=device,
                                                 target_class=target)

    sigmas = build_sigma_schedule(50)

    model = model_config['class'].to(device)
    model.load_state_dict(torch.load(f'../model_classes/models/{model_config["file"]}', map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    image, intermediate_images = sampling_pipeline(model, next(iter(dl.train))[0], sigmas, info.sigma_data, device=device, target=target,
                                                   conditional=model_config['conditional'])

    if evaluate:
        real_folder = "../images/fid_evaluation/real_images"
        generated_folder = "../images/fid_evaluation/generated_images"
        save_generated_images(image, generated_folder, "generated")

        for y, label in dl.train:
            save_generated_images(y, real_folder, "real")
            break

        fid = fid_score.calculate_fid_given_paths([real_folder, generated_folder], batch_size=32, device=device, dims=2048)
        print(f"FID score: {fid:.2f}")

    save_grid_image(image, '../images/outputs/loaded_model_images')
    animate_denoising([img.cpu() for img in intermediate_images], save_path="../images/outputs/loaded_model_denoising_steps")


if __name__ == '__main__':
    evaluate = True
    model_config = MODEL_CONFIGS['ModelClassConditional4']

    print(model_config['class'])
    run_sampling(model_config, evaluate=evaluate)
