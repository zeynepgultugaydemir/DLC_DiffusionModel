from model_classes.model import Model
from model_classes.model_class_conditional import ModelClassConditional
from model_classes.model_conditional_batch_norm import ModelConditionalBatchNorm
from utils.data_loaders import load_dataset_and_make_dataloaders
from utils.pipeline_utils import get_device, build_sigma_schedule, sampling_pipeline, training_pipeline
from utils.workflow_utils import animate_denoising, plot_and_save_losses, save_grid_image, save_model, map_name_to_class


def run_training(model_type, dataset_name, batch_size, target=None, demo=True):
    if model_type == 'base':
        model = Model(**model_kwargs)
    elif model_type == 'noisecond':
        model = ModelConditionalBatchNorm(**model_kwargs)
    elif model_type == 'classcond':
        model = ModelClassConditional(**model_kwargs)
        target = target

    device = get_device()
    model.to(device)
    dl, info = load_dataset_and_make_dataloaders(dataset_name=dataset_name, root_dir='../data', batch_size=batch_size, num_workers=0,
                                                 pin_memory=device)

    train_loss = training_pipeline(model, dataset_name=dataset_name, epochs=5, batch_size=batch_size, device=device, demo=demo)
    val_loss = training_pipeline(model, dataset_name=dataset_name, epochs=5, batch_size=batch_size, validation=True, device=device, demo=demo)

    sigmas = build_sigma_schedule(50, sigma_max=50)
    image, intermediate_images = sampling_pipeline(model, next(iter(dl.train))[0], sigmas, info.sigma_data, target=target, device=device)
    model.to(device)

    plot_and_save_losses(train_loss, val_loss, '../images/outputs/training_losses')
    save_grid_image(image, '../images/outputs/trained_model_images')
    animate_denoising([img.cpu() for img in intermediate_images], save_path="../images/outputs/trained_model_denoising_steps", interval=50)
    save_model(model, f"../model_classes/models/demo_model_{dataset_name}.pth")


if __name__ == '__main__':
    demo = True  # Set to True during presentation
    dataset_name = 'FashionMNIST'  # Choose FashionMNIST or CelebA
    model_type = 'classcond'  # Choose 'base', 'classcond' or 'noisecond' (CelebA cannot be run with classcond model)
    target = 'jacket'  # Provide the desired target class if you've chosen FashionMNIST on class conditioned model
    batch_size = 8
    base_kwargs = {
        'image_channels': 1,  # Set to 3 if CelebA, 1 if FashionMNIST
        'nb_channels': 8,
        'num_blocks': 1,
        'cond_channels': 8
    }

    if model_type == 'classcond':
        model_kwargs = {**base_kwargs, 'num_classes': 10}
        target = map_name_to_class(target)
    else:
        model_kwargs = {**base_kwargs}
        target = None

    run_training(model_type=model_type, dataset_name=dataset_name, batch_size=batch_size, target=target, demo=demo)
