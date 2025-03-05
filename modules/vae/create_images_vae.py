from diffusers import AutoencoderKL
import torch
from PIL import Image
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
import os

import hydra
from omegaconf import DictConfig

# Add the dataloader directory to the Python path
sys.path.append(os.path.join(os.getcwd(), "dataloader"))
import washu_dataloader as wu_dl


def load_trained_vae(device, model_name):
    # Load the trained VAE model
    model_path = f"./models/{model_name}"
    vae = AutoencoderKL.from_pretrained(model_path).to(device)
    vae.eval()  # Set to evaluation mode
    return vae

# Scale each channel to [0, 1]. Avoid division by zero if all values are the same
def scale_each_channel(tensor):
        
    min_vals = tensor.amin(dim=(1, 2), keepdim=True)  # Use amin for multi-dim min
    max_vals = tensor.amax(dim=(1, 2), keepdim=True)  # Use amax for multi-dim max

    return torch.where(
        max_vals == min_vals, 
        torch.ones_like(tensor),  
        (tensor - min_vals) / (max_vals - min_vals))

#takes a tensor and return an image of the components concatenated horizontally
def image_as_grid(tensor, dataset, mask=None):
    tensor = dataset.denormalize(tensor.detach())
    tensor = scale_each_channel(tensor)
    if(mask is not None): tensor = tensor * mask
    np_array = tensor.cpu().numpy()
    stacked_images = np.hstack(np_array)
    stacked_images = (stacked_images * 255).astype(np.uint8)
    img_pil = Image.fromarray(stacked_images, mode="L")
    return img_pil

def save_generated_images(images, mask, save_dir):
    mask = mask.cpu().numpy()  # Convert mask to NumPy
    os.makedirs(save_dir, exist_ok=True)

    num_outcomes = images.shape[1]  # Number of outcome variables

    # Remove batch dimension from mask
    mask = mask[0]  

    images = denormalize(images)  # Denormalize images

    for outcome_idx in range(num_outcomes):  
        outcome_dir = os.path.join(save_dir, f"outcome_{outcome_idx}")
        os.makedirs(outcome_dir, exist_ok=True)

        for i, img in enumerate(images.cpu().numpy()):
            img = img[outcome_idx]

            # Avoid division by zero if all values are the same
            img_min, img_max = img.min(), img.max()
            if img_max != img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.ones((128, 256), dtype=float)
            img = (img * 255).astype(np.uint8)

            # Apply mask
            masked_img = np.multiply(img, mask[outcome_idx].astype(np.uint8))

            masked_img_pil = Image.fromarray(masked_img, mode="L")
            masked_img_pil.save(os.path.join(outcome_dir, f"masked_sample_{i}.png"))
            img_pil = Image.fromarray(img, mode="L")
            img_pil.save(os.path.join(outcome_dir, f"sample_{i}.png"))

    print(f"Generated images saved in '{save_dir}' under separate folders for each outcome variable.")

def generate_samples_from_noise(vae, mask, device, batch_size, latent_dim, model_name):
    # Sample random latent vectors
    latent_shape = (batch_size, *latent_dim)
    random_latents = torch.randn(latent_shape).to(device)
    
    # Decode into images
    with torch.no_grad():
        generated_images = vae.decode(random_latents).sample  # shape: (batch, 2, 128, 256)
    
    save_dir=f"./experiments/vae_model_performance/{model_name}_images/generated_samples"
    save_generated_images(generated_images, mask, save_dir)

def reconstruct_samples_via_vae(vae, device, dataloader, model_name):
    # Obtain a batch from the dataloader
    batch, mask = next(iter(dataloader))
    real_images = batch.to(device)
    latent_shape = None
    # Encode and decode to reconstruct images
    with torch.no_grad():
        posterior = vae.encode(real_images)
        latents = posterior.latent_dist.sample()
        latent_shape = latents.shape
        reconstructed_images = vae.decode(latents).sample
    
    save_dir=f"./experiments/vae_model_performance/{model_name}_images/reconstructed_samples"
    save_generated_images(reconstructed_images, mask, save_dir)
    print(f"Reconstructed images saved in '{save_dir}'")
    return latent_shape

def save_images_from_dataset(dataloader, save_dir="./experiments/dataset_samples_normalized"):
    # Save real images from the dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, mask = next(iter(dataloader))
    #batch = denormalize(batch.to(device))
    real_images = batch.cpu().numpy()
    os.makedirs(save_dir, exist_ok=True)
    num_outcomes = real_images.shape[1]

    for outcome_idx in range(num_outcomes):
        outcome_dir = os.path.join(save_dir, f"outcome_{outcome_idx}")
        os.makedirs(outcome_dir, exist_ok=True)

        for i, img in enumerate(real_images):  
            #print(f"Image {i} - Min: {img.min()}, Max: {img.max()}")
            img = img[outcome_idx]
            img = (img - img.min()) / (img.max() - img.min())
            img = (img * 255).astype(np.uint8)

            img_pil = Image.fromarray(img, mode="L")
            img_pil.save(os.path.join(outcome_dir, f"real_{i}.png"))

    print(f"Real images saved in '{save_dir}' under separate folders for each outcome variable.")

@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    dataset = wu_dl.initialize_dataset(cfg.root_dir, cfg.grid_size, cfg.components)      
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,   #256,512 image size on "simple" vae allowed batch size of 6. But we want to train and experiment faster
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    first_batch = next(iter(loader))
    padded = torch.nan_to_num(first_batch[0], nan=0.0)
    image = image_as_grid(padded, dataset)
    #print("Image shape: ", image.size)
    image.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    main()
    # Load the trained VAE
    #model_name = "sd_vae"
    #vae = load_trained_vae(device, model_name)
    #dataloader = initialize_data_loader(components = ["PM25", "BC"], batch_size=6, shuffle=False, img_size=(128, 256))

    #_, mask = next(iter(dataloader))

    # Reconstruct samples via VAE
    #latent_shape = reconstruct_samples_via_vae(vae,device, dataloader, model_name=model_name)
    #print("Latent shape: ", latent_shape)

    # Generate samples from random noise
    #generate_samples_from_noise(vae, device=device,mask=mask, batch_size=6,latent_dim=latent_shape[1:], model_name=model_name)
    
    # Save real images from dataset -- dont need to do it every time
    #save_images_from_dataset(dataloader)

    