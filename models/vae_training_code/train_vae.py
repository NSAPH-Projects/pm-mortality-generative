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


import sys, os
sys.path.append(os.path.join(os.getcwd(), "dataloader"))
from washu_dataloader import initialize_dataset

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

def print_vae_info():
    myparam = vae.named_parameters();
    print("Initial Model Weights:")
    for name, param in vae.named_parameters():
        if param.requires_grad:
            print(f"{name}: Mean={param.data.mean().item()}, Std={param.data.std().item()}")
    print(f"config: {vae.config}")

def train_vae(cfg :DictConfig , vae, vae_name, data_loader):
    
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, 
                config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True
    ))

    vae_optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    for epoch in tqdm(range(cfg.num_epochs), desc='Epochs'):
        vae.train()
        
        epoch_loss = 0  # To accumulate loss over each epoch
        for batch in tqdm(data_loader, desc='Batches', leave=False):  # tqdm will show progress for batches within each epoch
            
            batch = batch.to(device)
            mask = ~torch.isnan(batch)

            # I think we should pad with the minimum element of the current batch instead of zero. Because what was zero previously is a negative number after normalization
            batch_padded = torch.nan_to_num(batch, nan=0.0)
            
            vae_optimizer.zero_grad()
            
            # VAE forward pass
            posterior = vae.encode(batch_padded)
            latents = posterior.latent_dist.sample()
            #print("latent shape: ", latents.shape)
            reconstructed = vae.decode(latents).sample
            
            # Compute VAE loss
            #print the number of nan values and total number of values in the batch
            #print("Number of nan values in batch: ", mask.sum() , " Total number of values in batch: ", batch.numel())
            recon_loss = F.mse_loss(reconstructed[mask], batch[mask])
            kl_loss = posterior.latent_dist.kl().mean()
            vae_loss = recon_loss + cfg.kl_weight * kl_loss

            vae_loss.backward()
            vae_optimizer.step()
            
            epoch_loss += vae_loss.item()
        
        average_loss = epoch_loss / len(data_loader)
        wandb.log({"loss": average_loss})
        #print(f"VAE Epoch {epoch+1}, Average Loss: {average_loss:.4f}")

    save_path = f"./models/{vae_name}"
    os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
    vae.save_pretrained(f"./models/{vae_name}")

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

@hydra.main(config_path="/Users/oahmet/Projects/pm-mortality-generative/conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    dataset = initialize_dataset(cfg.root_dir, cfg.grid_size, cfg.components)      
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,   #256,512 image size on "simple" vae allowed batch size of 6. But we want to train and experiment faster
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    vae_name = "sd_vae"
    vae = get_vae(vae_name, device, len(cfg.components))

    train_vae(cfg, vae, vae_name, loader)
    print("Training complete.")

if __name__ == "__main__":
    main()