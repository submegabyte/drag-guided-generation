import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from cc_ddpm import *

# Training loop
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleDenoiser(num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training
epochs = 5
for epoch in range(epochs):
    for x0, labels in dataloader:
        x0 = x0.to(device)
        labels = labels.to(device)
        t = torch.randint(0, T, (x0.shape[0],), device=device)  # Random timesteps
        x_t, noise = forward_diffusion(x0, t)
        noise_pred = model(x_t, t, labels)  # Condition on class label
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}: Loss = {loss.item()}")

print("Training complete!")

# Ensure the 'models' folder exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Save the model with the same name as the script
model_path = os.path.join(models_dir, f"{model_name}.pth")
torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")
