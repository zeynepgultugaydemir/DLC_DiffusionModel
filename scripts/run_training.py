from model_classes.model import Model
from utils.data_loaders import load_dataset_and_make_dataloaders
from utils.pipeline_utils import get_device, build_sigma_schedule, \
    compute_c_functions, sampling_pipeline, training_pipeline
from utils.workflow_utils import animate_denoising, plot_and_save_losses, save_grid_image

device = get_device()
print(device)

model = Model(image_channels=1, nb_channels=8, num_blocks=1, cond_channels=8).to(device)
model.to(device)

dl, info = load_dataset_and_make_dataloaders(dataset_name='FashionMNIST', root_dir='../data', batch_size=8, num_workers=0, pin_memory=device)
for y, label in dl.train:
    images = y
    print(y.shape)
    print(label.shape)
    break

sigmas = build_sigma_schedule(50, sigma_max=50)
c_funcs = compute_c_functions(sigmas, info.sigma_data)

train_loss = training_pipeline(model, epochs=3, device=device, demo=True)
val_loss = training_pipeline(model, epochs=3, validation=True, device=device, demo=True)

# save_model(model, "../model_classes/models/demo_model.pth")

image, intermediate_images = sampling_pipeline(model, images, sigmas, info.sigma_data, device=device)
model = model.to(device)

plot_and_save_losses(train_loss, val_loss, 'trained_model_losses')
save_grid_image(image, 'trained_model_images')

animate_denoising([img.cpu() for img in intermediate_images], save_path="trained_model_denoising", interval=50)
