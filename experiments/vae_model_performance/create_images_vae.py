from diffusers import AutoencoderKL
import torch
from PIL import Image
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
import os

# Add the dataloader directory to the Python path
sys.path.append(os.path.join(os.getcwd(), "dataloader"))
#sys.path.append("/n/dominici_lab/lab/projects/pm-mortality-generative/ahmet/pm-mortality-generative/dataloader")
#sys.path.append("/Users/oahmet/Projects/pm-mortality-generative/dataloader")
# Print the Python path to verify
print("Python path:", sys.path)

from climate_data_handling import initialize_data_loader

def load_trained_vae(device, model_name):
    # Load the trained VAE model
    model_path = f"./models/{model_name}"
    vae = AutoencoderKL.from_pretrained(model_path).to(device)
    vae.eval()  # Set to evaluation mode
    return vae

def save_generated_images(images, save_dir):
    # Convert to NumPy for visualization and save
    os.makedirs(save_dir, exist_ok=True)
    num_outcomes = images.shape[1]  # Number of outcome variables

    for outcome_idx in range(num_outcomes):  
        outcome_dir = os.path.join(save_dir, f"outcome_{outcome_idx}")
        os.makedirs(outcome_dir, exist_ok=True)

        for i, img in enumerate(images.cpu().numpy()):
            img = img[outcome_idx]
            img = (img - img.min()) / (img.max() - img.min())
            img = (img * 255).astype(np.uint8)

            img_pil = Image.fromarray(img, mode="L")
            img_pil.save(os.path.join(outcome_dir, f"sample_{i}.png"))

    print(f"Generated images saved in '{save_dir}' under separate folders for each outcome variable.")


def generate_samples_from_noise(vae, device, batch_size=6, latent_dim=(4, 128, 256), save_dir="./experiments/vae_model_performance/sd_vae_images/generated_samples"):
    # Sample random latent vectors
    latent_shape = (batch_size, *latent_dim)
    random_latents = torch.randn(latent_shape).to(device)
    
    # Decode into images
    with torch.no_grad():
        generated_images = vae.decode(random_latents).sample  # shape: (batch, 2, 128, 256)
        
    save_generated_images(generated_images, save_dir)

def reconstruct_samples_via_vae(vae, dataloader, device, save_dir="./experiments/vae_model_performance/sd_vae_images/reconstructed_samples"):
    # Obtain a batch from the dataloader
    real_images = next(iter(dataloader)).to(device)
    
    # Encode and decode to reconstruct images
    with torch.no_grad():
        posterior = vae.encode(real_images)
        latents = posterior.latent_dist.sample()
        print(latents.shape)
        reconstructed_images = vae.decode(latents).sample
    
    save_generated_images(reconstructed_images, save_dir)
    print(f"Reconstructed images saved in '{save_dir}'")

def save_images_from_dataset(dataloader, save_dir="./experiments/dataset_samples"):
    # Save real images from the dataset
    real_images = next(iter(dataloader)).cpu().numpy()
    os.makedirs(save_dir, exist_ok=True)
    num_outcomes = real_images.shape[1]

    for outcome_idx in range(num_outcomes):
        outcome_dir = os.path.join(save_dir, f"outcome_{outcome_idx}")
        os.makedirs(outcome_dir, exist_ok=True)

        for i, img in enumerate(real_images):  
            print(f"Image {i} - Min: {img.min()}, Max: {img.max()}")
            img = img[outcome_idx]
            img = (img - img.min()) / (img.max() - img.min())
            img = (img * 255).astype(np.uint8)

            img_pil = Image.fromarray(img, mode="L")
            img_pil.save(os.path.join(outcome_dir, f"real_{i}.png"))

    print(f"Real images saved in '{save_dir}' under separate folders for each outcome variable.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    # Load the trained VAE
    vae = load_trained_vae(device, "sd_vae")
    dataloader = initialize_data_loader(components = ["PM25", "BC"], years = [2000], batch_size=6, shuffle=False, img_size=(128, 256))
    
    # Generate samples from random noise
    #generate_samples_from_noise(vae,batch_size=6,latent_dim=(4, 64, 128), device=device)
    
    # Reconstruct samples via VAE
    reconstruct_samples_via_vae(vae, dataloader, device)
    
    # Save real images from dataset -- dont need to do it every time
    #save_images_from_dataset(dataloader)