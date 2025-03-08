## https://chatgpt.com/share/67cbf4db-716c-800c-97e4-57444b099efb

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

## enable cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the noise schedule
T = 1000  # Total timesteps
betas = torch.linspace(1e-4, 0.02, T).to(device)  # Linear schedule
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Forward Diffusion Process (same)
def forward_diffusion(x0, t, noise=None):
    """Adds Gaussian noise to x0 at timestep t."""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_cumprod_t = (alphas_cumprod[t] ** 0.5)[:, None, None, None]
    sqrt_one_minus_alpha_cumprod_t = ((1 - alphas_cumprod[t]) ** 0.5)[:, None, None, None]
    return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise, noise

# Class-Conditioned Diffusion Model (UNet-like)
class SimpleDenoiser(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.fc_class_embed = nn.Embedding(num_classes, 64)  # Class embedding
        self.model = nn.Sequential(
            nn.Conv2d(1 + 64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t, class_idx):
        # Concatenate class embedding to the image
        class_embed = self.fc_class_embed(class_idx).view(-1, 64, 1, 1)
        class_embed = class_embed.expand(-1, -1, x.size(2), x.size(3))  # Match image dimensions
        x = torch.cat([x, class_embed], dim=1)  # Concatenate class with the image
        return self.model(x)


# Get the script's filename without extension
model_name = os.path.splitext(os.path.basename(__file__))[0]