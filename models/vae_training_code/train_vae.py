from diffusers import AutoencoderKL
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys, os
sys.path.append(os.path.join(os.getcwd(), "dataloader"))
from climate_data_handling import initialize_data_loader

def simple_vae(device, num_channels):
    # Initialize the VAE model from scratch
    vae = AutoencoderKL(
        in_channels=num_channels,  # not RGB images
        out_channels=num_channels,  # not RGB images
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(64, 128),  # Customize as needed
        latent_channels=4  # Latent space size, adjust as needed
    )
    vae = vae.to(device)  # Move to GPU if available
    return vae

def stable_diffusion_vae(device, num_channels):
    # Load the trained VAE model
    pretrained_model_name = "CompVis/stable-diffusion-v1-4"
    config = AutoencoderKL.load_config(pretrained_model_name, subfolder="vae")

    #fix in and out channels to match our data
    config["in_channels"] = num_channels
    config["out_channels"] = num_channels

    # Initialize a new, untrained AutoencoderKL model with the loaded configuration
    vae = AutoencoderKL.from_config(config)
    vae = vae.to(device)  # Move to GPU if available
    return vae

def print_vae_info():
    myparam = vae.named_parameters();
    print("Initial Model Weights:")
    for name, param in vae.named_parameters():
        if param.requires_grad:
            print(f"{name}: Mean={param.data.mean().item()}, Std={param.data.std().item()}")
    print(f"config: {vae.config}")

def train_vae(vae, vae_name, data_loader, num_epochs=30, lr=1e-4, kl_weight=0.1):
    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        vae.train()
        
        epoch_loss = 0  # To accumulate loss over each epoch
        for batch, mask in tqdm(data_loader, desc='Batches', leave=False):  # tqdm will show progress for batches within each epoch
            batch = batch.to(device)
            vae_optimizer.zero_grad()
            
            # VAE forward pass
            posterior = vae.encode(batch)
            latents = posterior.latent_dist.sample()
            #print("latent shape: ", latents.shape)
            reconstructed = vae.decode(latents).sample
            
            # Compute VAE loss
            #print the number of nan values and total number of values in the batch
            #print("Number of nan values in batch: ", mask.sum() , " Total number of values in batch: ", batch.numel())
            recon_loss = F.mse_loss(reconstructed[mask], batch[mask])
            kl_loss = posterior.latent_dist.kl().mean()
            vae_loss = recon_loss + kl_weight * kl_loss

            vae_loss.backward()
            vae_optimizer.step()
            
            epoch_loss += vae_loss.item()
        
        average_loss = epoch_loss / len(data_loader)
        print(f"VAE Epoch {epoch+1}, Average Loss: {average_loss:.4f}")

    vae.save_pretrained(f"./models/{vae_name}")
    print("VAE model saved.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    #try different values of img_size and find the best one
    #256,512 image size on "simple" vae allowed batch size of 6. But we want to train and experiment faster
    components = ["PM25", "BC"]
    dataloader = initialize_data_loader(components = components, batch_size=12, shuffle=True, img_size=(128, 256))

    vae = simple_vae(device, num_channels=len(components))
    print("VAE model initialized.")
  
    train_vae(vae, "simple_vae", dataloader, num_epochs=170, lr=1e-4, kl_weight=0.1)
    print("Training complete.")