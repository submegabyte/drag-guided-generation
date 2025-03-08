## https://chatgpt.com/share/67cb6b10-14cc-800c-a045-d68d6dbb4568

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

## enable cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
T = 300  # Number of diffusion steps
beta = torch.linspace(1e-4, 0.02, T).to(device)  # Noise schedule
alpha = 1 - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# Simple forward diffusion process
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
    return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise, noise

# Simple U-Net-inspired denoiser
class SimpleDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1)
        )

    def forward(self, x, t):
        return self.net(x)

# Training setup
model = SimpleDenoiser().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Load dataset (MNIST for simplicity)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# Training loop
epochs = 100
for epoch in range(epochs):
    loss = 0
    for i, (x0, _) in enumerate(dataloader):
        x0 = x0.to(device)
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        xt, noise = forward_diffusion(x0, t)
        noise_pred = model(xt, t)
        loss_current = loss_fn(noise_pred, noise)
        loss += loss_current
    
        print(f"Epoch {epoch+1}, datapoint {i} Loss: {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Sampling (reverse diffusion)
@torch.no_grad()
def sample():
    x = torch.randn((1, 1, 28, 28), device=device)
    for t in reversed(range(T)):
        z = torch.randn_like(x) if t > 0 else 0
        noise_pred = model(x, torch.tensor([t], device=device))
        x = (x - beta[t] * noise_pred) / torch.sqrt(alpha[t]) + torch.sqrt(beta[t]) * z
    return x

# Save the generated image
sampled_image = sample().cpu().squeeze().numpy()
plt.imshow(sampled_image, cmap="gray")
plt.axis("off")  # Remove axes for a cleaner image
# plt.savefig("generated_image.png", bbox_inches="tight", pad_inches=0)  # Save image
# plt.show()

# Ensure the 'results' folder exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Save the generated image in the 'results' folder
save_path = os.path.join(results_dir, "generated_image.png")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
print(f"Image saved to {save_path}")

# Get the script's filename without extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Ensure the 'models' folder exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Save the model with the same name as the script
model_path = os.path.join(models_dir, f"{script_name}.pth")
torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")
