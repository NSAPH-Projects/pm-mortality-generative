from diffusers import AutoencoderKL
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data_loader import ClimateDataset

def initialize_vae(device, num_channels):
    # Initialize the VAE model from scratch
    vae = AutoencoderKL(
        in_channels=num_channels,  # not RGB images
        out_channels=num_channels,  # not RGB images
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(64, 128),  # Customize as needed
        latent_channels=4  # Latent space size, adjust as needed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    vae = vae.to(device)  # Move to GPU if available
    return vae

def print_vae_info():
    myparam = vae.named_parameters();
    print("Initial Model Weights:")
    for name, param in vae.named_parameters():
        if param.requires_grad:
            print(f"{name}: Mean={param.data.mean().item()}, Std={param.data.std().item()}")
    print(f"config: {vae.config}")

def initialize_data_loader(components, years, batch_size, img_size=(128, 256)):
    # Load your custom dataset
    root = "./data/climate-monthly/netcdf"
    transformations = transforms.Resize(img_size)
    dataset = ClimateDataset(root, components, years, transformations=transformations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_vae(vae, data_loader, num_epochs=30, lr=1e-4, kl_weight=0.1):
    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        vae.train()
        
        epoch_loss = 0  # To accumulate loss over each epoch
        for batch in tqdm(data_loader, desc='Batches', leave=False):  # tqdm will show progress for batches within each epoch
            batch = batch.to(device)
            vae_optimizer.zero_grad()
            
            # VAE forward pass
            posterior = vae.encode(batch)
            latents = posterior.latent_dist.sample()
            #print("latent shape: ", latents.shape)
            reconstructed = vae.decode(latents).sample
            
            # Compute VAE loss
            recon_loss = F.mse_loss(reconstructed, batch)
            kl_loss = posterior.latent_dist.kl().mean()
            vae_loss = recon_loss + kl_weight * kl_loss

            vae_loss.backward()
            vae_optimizer.step()
            
            epoch_loss += vae_loss.item()
        
        average_loss = epoch_loss / len(data_loader)
        print(f"VAE Epoch {epoch+1}, Average Loss: {average_loss:.4f}")

    vae.save_pretrained("./models/custom_vae")
    print("VAE model saved.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    #prepare the data loader
    components = ["PM25", "BC"]
    #list years between 2000 and 2017, (many outcomes don't have 2018 data)
    years = list(range(2000, 2018))
    
    #try different values of img_size and find the best one 
    dataloader = initialize_data_loader(components, years, batch_size=6, img_size=(256, 512))

    vae = initialize_vae(device, num_channels=len(components))
    print("VAE model initialized.")
  
    train_vae(vae,dataloader, num_epochs=30, lr=1e-4)
    print("Training complete.")