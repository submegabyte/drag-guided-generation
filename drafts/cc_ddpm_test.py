import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from cc_ddpm import *

## enable cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure the 'models' folder exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, f"{model_name}.pth")

# Load the model from the saved file
model = SimpleDenoiser().to(device)
if os.path.exists(model_path):  # Check if the file exists before loading
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {model_path}")
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

# DDIM Sampling (Conditioned on class label)
def ddim_sample(model, shape, class_idx, ddim_steps=50):
    """DDIM sampling from pure noise with class conditioning"""
    device = next(model.parameters()).device
    x_t = torch.randn(shape, device=device)  # Start from pure noise
    step_size = T // ddim_steps
    ts = list(reversed(range(0, T, step_size)))

    for i in range(len(ts) - 1):
        t = torch.full((shape[0],), ts[i], device=device, dtype=torch.long)
        t_next = torch.full((shape[0],), ts[i + 1], device=device, dtype=torch.long)

        alpha_t = alphas_cumprod[t][:, None, None, None]
        alpha_t_next = alphas_cumprod[t_next][:, None, None, None]
        sqrt_alpha_t = alpha_t ** 0.5
        sqrt_alpha_t_next = alpha_t_next ** 0.5
        sqrt_one_minus_alpha_t = (1 - alpha_t) ** 0.5

        noise_pred = model(x_t, t, class_idx)  # Predict noise conditioned on class
        x0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t  # Reconstruct x0

        x_t = sqrt_alpha_t_next * x0_pred + (1 - alpha_t_next) ** 0.5 * noise_pred  # Update x_t

    return x_t

# Generate samples using DDIM
model.eval()
with torch.no_grad():
    sampled_images = ddim_sample(model, (16, 1, 28, 28), class_idx=torch.randint(0, 10, (16,)).to(device), ddim_steps=50)

# Display results
fig, axes = plt.subplots(2, 8, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(sampled_images[i].cpu().squeeze(), cmap="gray")
    ax.axis("off")
# plt.show()

# Ensure the 'results' folder exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Save the generated image in the 'results' folder
save_path = os.path.join(results_dir, "generated_image.png")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
print(f"Image saved to {save_path}")
