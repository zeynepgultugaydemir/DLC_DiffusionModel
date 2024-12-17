import torch
import torch.nn as nn
import torch.nn.functional as F

from model_classes.noise_embedding import NoiseEmbedding


class ModelConditionalBatchNorm(nn.Module):
    def __init__(self, image_channels: int, nb_channels: int, num_blocks: int, cond_channels: int) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding(cond_channels)  # Noise embedding module
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)

        # Noise-conditioned residual blocks
        self.blocks = nn.ModuleList([ResidualConditionalBatchNorm(nb_channels, cond_channels) for _ in range(num_blocks)])

        # Output convolution with zero-initialized weights for stability
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_emb(c_noise)  # Implement noise embedding
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x, cond)
        return self.conv_out(x)


class ResidualConditionalBatchNorm(nn.Module):
    def __init__(self, nb_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels, affine=False)  # Disable default learnable affine params
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(nb_channels, affine=False)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)

        # Learnable conditional affine parameters (scale and shift)
        self.cond_scale1 = nn.Linear(cond_channels, nb_channels)
        self.cond_shift1 = nn.Linear(cond_channels, nb_channels)
        self.cond_scale2 = nn.Linear(cond_channels, nb_channels)
        self.cond_shift2 = nn.Linear(cond_channels, nb_channels)

        # Zero-initialize second convolution weights for stability
        nn.init.zeros_(self.conv2.weight)
        self.c_skip = nn.Parameter(torch.ones(1, nb_channels, 1, 1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # Conditional scale and shift for the first BatchNorm
        scale1 = self.cond_scale1(cond).view(-1, x.size(1), 1, 1)
        shift1 = self.cond_shift1(cond).view(-1, x.size(1), 1, 1)
        y = self.conv1(F.relu(self.norm1(x)) * scale1 + shift1)

        # Conditional scale and shift for the second BatchNorm
        scale2 = self.cond_scale2(cond).view(-1, x.size(1), 1, 1)
        shift2 = self.cond_shift2(cond).view(-1, x.size(1), 1, 1)
        y = self.conv2(F.relu(self.norm2(y)) * scale2 + shift2)

        # Combine with skip connection
        return self.c_skip * x + y
