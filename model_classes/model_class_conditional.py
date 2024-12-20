import torch
import torch.nn as nn
import torch.nn.functional as F

from model_classes.noise_embedding import NoiseEmbedding


class ModelClassConditional(nn.Module):
    def __init__(self, image_channels: int, nb_channels: int, num_blocks: int, cond_channels: int, num_classes: int = None) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding(cond_channels)

        # Mapping the class vector to the same dimensionality as the noise embedding
        self.class_emb = nn.Linear(num_classes, cond_channels) if num_classes is not None else None
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualClassConditional(nb_channels, cond_channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor, class_vector=None) -> torch.Tensor:
        noise_embedding = self.noise_emb(c_noise)

        # Combine noise and class embeddings if class_vector is provided
        if class_vector is not None:
            class_embedding = self.class_emb(class_vector)
            cond = noise_embedding + class_embedding
        else:
            cond = noise_embedding  # Use only the noise embedding if no class conditioning

        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x, cond)
        return self.conv_out(x)


class ResidualClassConditional(nn.Module):
    def __init__(self, nb_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels, affine=False)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(nb_channels, affine=False)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)

        self.cond_scale1 = nn.Linear(cond_channels, nb_channels)
        self.cond_shift1 = nn.Linear(cond_channels, nb_channels)
        self.cond_scale2 = nn.Linear(cond_channels, nb_channels)
        self.cond_shift2 = nn.Linear(cond_channels, nb_channels)

        nn.init.zeros_(self.conv2.weight)
        self.c_skip = nn.Parameter(torch.ones(1, nb_channels, 1, 1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale1 = self.cond_scale1(cond).view(-1, x.size(1), 1, 1)
        shift1 = self.cond_shift1(cond).view(-1, x.size(1), 1, 1)
        y = self.conv1(F.relu(self.norm1(x)) * scale1 + shift1)

        scale2 = self.cond_scale2(cond).view(-1, x.size(1), 1, 1)
        shift2 = self.cond_shift2(cond).view(-1, x.size(1), 1, 1)
        y = self.conv2(F.relu(self.norm2(y)) * scale2 + shift2)

        return self.c_skip * x + y
