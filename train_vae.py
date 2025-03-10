from diffusers import AutoencoderKL
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import hydra
from omegaconf import DictConfig
import wandb
import omegaconf
import numpy as np
from modules.vae.create_images_vae import image_as_grid
from datetime import datetime


import sys, os
#sys.path.append(os.path.join(os.getcwd(), "dataloader"))

import dataloader.washu_dataloader as wu_dl

import dataloader.climate_data_handling as ch
from modules.vae.create_images_vae import save_generated_images

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

    config["in_channels"] = num_channels
    config["out_channels"] = num_channels

    vae = AutoencoderKL.from_config(config)
    vae = vae.to(device)
    return vae

def print_vae_info(vae):
    myparam = vae.named_parameters();
    print("Initial Model Weights:")
    for name, param in vae.named_parameters():
        if param.requires_grad:
            print(f"{name}: Mean={param.data.mean().item()}, Std={param.data.std().item()}")
    print(f"config: {vae.config}")

def fill_nan_with_min(batch):
    B, C, _, _ = batch.shape

    nan_mask = torch.isnan(batch)

    # Replace NaNs with a large positive value temporarily
    batch_without_nan = batch.clone()
    batch_without_nan[nan_mask] = float('inf')

    # Compute the minimum per image, ignoring the temporarily replaced values
    min_vals = batch_without_nan.view(B, C, -1).min(dim=2, keepdim=True)[0].view(B, C, 1, 1)

    # Replace NaNs with the corresponding min value
    batch[nan_mask] = min_vals.expand_as(batch)[nan_mask]

    return batch

def fill_nan_with_mean_div_std(batch, means, stds):
     #not implemented yet
    return batch

def train_vae(cfg :DictConfig , vae, vae_name, data_loader, dataset):
    
    today_date = datetime.today().strftime('%Y-%m-%d')

    # Initialize Weights & Biases with a unique run name
    run_name = f"run_{today_date}_{cfg.run_id}"

    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=run_name,
                config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True
    ))

    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    for epoch in tqdm(range(cfg.num_epochs), desc='Epochs'):
        vae.train()
        
        #save the input and reconstructed images to wandb at the beginning of each epoch
        first_batch = True

        epoch_loss = 0  # To accumulate loss over each epoch
        epoch_kl_loss = 0
        epoch_recon_loss = 0

        for batch in tqdm(data_loader, desc='Batches', leave=False):  # tqdm will show progress for batches within each epoch
            
            batch = batch.to(device)
            mask = ~torch.isnan(batch)

            batch_padded = fill_nan_with_min(batch) #this is same as first filling nan with zero and then normalizing. But we first normalize
            
            vae_optimizer.zero_grad()
            
            # VAE forward pass
            posterior = vae.encode(batch_padded)
            latents = posterior.latent_dist.sample()
            #print("latent shape: ", latents.shape)
            reconstructed = vae.decode(latents).sample

            if(first_batch):
                #save the input and reconstructed images to wandb
                input_image = image_as_grid(batch_padded[0], dataset) #dataset is passed to be able to denormalize
                reconstructed_image = image_as_grid(reconstructed[0], dataset)
                masked_rc_image = image_as_grid(reconstructed[0], dataset, mask[0])
                wandb.log({"input_images": [wandb.Image(input_image)],
                           "masked_reconstructed_images": [wandb.Image(masked_rc_image)],
                            "reconstructed_images": [wandb.Image(reconstructed_image)] })
                first_batch = False
        
            # Compute VAE loss
            #print the number of nan values and total number of values in the batch
            #print("Number of nan values in batch: ", mask.sum() , " Total number of values in batch: ", batch.numel())
            recon_loss = F.mse_loss(reconstructed[mask], batch[mask])
            kl_loss = posterior.latent_dist.kl().mean()
            vae_loss = recon_loss + cfg.kl_weight * kl_loss

            vae_loss.backward()
            vae_optimizer.step()
            
            epoch_loss += vae_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_recon_loss += recon_loss.item()
        
        average_loss = epoch_loss / len(data_loader)
        average_kl_loss = epoch_kl_loss / len(data_loader)
        average_recon_loss = epoch_recon_loss / len(data_loader)

        wandb.log({"loss": average_loss})
        wandb.log({"kl_loss": average_kl_loss})
        wandb.log({"recon_loss": average_recon_loss})

        #print(f"VAE Epoch {epoch+1}, Average Loss: {average_loss:.4f}")

    save_path = f"./models/{vae_name}/{run_name}"
    os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
    vae.save_pretrained(save_path)

    print("VAE model saved.")

def get_vae(vae_name, device, num_components):
    if(vae_name == "sd_vae"):
        vae = stable_diffusion_vae(device, num_channels=num_components)
    elif(vae_name == "simple_vae"):
        vae = simple_vae(device, num_channels=num_components)
    else: 
        print("Invalid vae name. Please use 'sd_vae' or 'simple_vae'")
        return
    print("VAE model initialized.")
    return vae

@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    dataset = wu_dl.initialize_dataset(cfg.root_dir, cfg.grid_size, cfg.components)      
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,   #256,512 image size on "simple" vae allowed batch size of 6. But we want to train and experiment faster
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    #loader = ch.initialize_data_loader(components = ["PM25", "BC"], batch_size=cfg.batch_size, shuffle=True, img_size=cfg.grid_size)

    if(cfg.load_pretrained):
        vae = AutoencoderKL.from_pretrained(f"./models/{cfg.vae_name}/{cfg.run_name}", use_safetensors = True).to(device)
    else:
        vae = get_vae(cfg.vae_name, device, len(cfg.components))

    train_vae(cfg, vae, cfg.vae_name, loader, dataset)
    print("Training complete.")

if __name__ == "__main__":
    main()