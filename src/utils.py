from diffusers import AutoencoderKL
import torch
import PIL
from PIL import Image
import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Union, List

import sys
import os

import hydra
from omegaconf import DictConfig

# Add the dataloader directory to the Python path
sys.path.append(os.path.join(os.getcwd(), "src/dataloader"))
import washu_dataloader as wu_dl


def load_trained_vae(device, model_name):
    # Load the trained VAE model
    model_path = f"./models/{model_name}"
    vae = AutoencoderKL.from_pretrained(model_path).to(device)
    vae.eval()  # Set to evaluation mode
    return vae

def fill_nan_with_min(batch, min_vals, nan_mask):
    min_vals = torch.FloatTensor(min_vals).to(batch.device)
    min_vals = min_vals.view(1, -1, 1, 1)  # Reshape to match (B, C, H, W) broadcasting
    filled_batch = torch.where(nan_mask, min_vals, batch)
    return filled_batch

def denormalize(tensor, means, stds):
    device = tensor.device  # Use the same device as the input tensor
    
    mean = torch.as_tensor(means, dtype=tensor.dtype, device=device).view(-1, 1, 1)  # (1, C, 1, 1)
    std = torch.as_tensor(stds, dtype=tensor.dtype, device=device).view(-1, 1, 1)    # (1, C, 1, 1)
    
    #if we want to denormalize a batch of images
    if(tensor.dim() == 4):
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean

# Scale each channel to [0, 1]. Avoid division by zero if all values are the same. Works for batched and unbatched tensors.
def scale_channels_to_one(tensor):
    min_vals = tensor.amin(dim=(-2, -1), keepdim=True)  # Use amin for multi-dim min
    max_vals = tensor.amax(dim=(-2, -1), keepdim=True)  # Use amax for multi-dim max

    return torch.where(
        max_vals == min_vals, 
        torch.ones_like(tensor),  
        (tensor - min_vals) / (max_vals - min_vals))

#takes a tensor and return a numpy array of the image of the components concatenated horizontally
def channels_seperated_image(tensor, means, stds, output="pil", mask=None)-> Union[List[PIL.Image.Image], np.ndarray]:
    b, c, h, w = tensor.shape
    tensor = denormalize(tensor.detach(), means, stds)
    tensor = scale_channels_to_one(tensor)
    if(mask is not None): tensor = tensor * mask
    np_array = tensor.cpu().numpy()
    images_in_row = np_array.np_array.transpose(0, 2, 1, 3).reshape(b, h, c * w)
    images_in_row = (images_in_row * 255).astype(np.uint8)
    if output == "pil":
        img_list = [Image.fromarray(img, mode="L") for img in images_in_row]
        return img_list
    return images_in_row

#this function is to create a pil image that can help us compare the ground truth and the generated images
def stacked_image(generated, groundtruth, means, stds, output='pil', mask=None):
    #check if the batch size of the generated and groundtruth images are 1
    assert generated.shape[0] == 1 and groundtruth.shape[0] == 1, "Batch size must be 1"
    
    generated = channels_seperated_image(generated, means, stds, output="numpy" , mask=mask)[0]
    groundtruth = channels_seperated_image(groundtruth, means, stds, output="numpy" , mask=mask)[0]
    stacked_image = np.concatenate((generated, groundtruth), axis=-1)
    if output == 'pil':
        return Image.fromarray(stacked_image, mode="L")
    return stacked_image

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

    