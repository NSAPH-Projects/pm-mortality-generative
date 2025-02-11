from diffusers import AutoencoderKL
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data_loader import ClimateDataset

# Initialize the VAE model from scratch
vae = AutoencoderKL(
    in_channels=2,  # not RGB images
    out_channels=2,  # not RGB images
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
    block_out_channels=(64, 128),  # Customize as needed
    latent_channels=4  # Latent space size, adjust as needed
)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

vae = vae.to(device)  # Move to GPU if available

# Define optimizer
optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4)

# Loss function
loss_fn = nn.MSELoss()  # Can be replaced with other options

# Define transformations for your custom dataset
transform = transforms.Compose([
    transforms.Resize((128, 256))])

# Load your custom dataset
root = "./data/climate-monthly/netcdf"
components = ["PM25", "BC"]
years = [2013, 2014, 2015, 2016]
transformations = transforms.Resize((128, 256))
dataset = ClimateDataset(root, components, years, transformations=transformations)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    vae.train()
    total_loss = 0

    for batch in dataloader:
        
        images = batch.to(device)
        optimizer.zero_grad()

        # Encode and Decode with VAE
        latents = vae.encode(images).latent_dist.sample()
        recon_images = vae.decode(latents).sample

        # Compute reconstruction loss
        loss = loss_fn(recon_images, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Save the trained VAE model
vae.save_pretrained("./models/custom_vae")
