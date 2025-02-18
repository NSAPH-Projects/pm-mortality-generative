from diffusers import AutoencoderKL

def print_model_config(model):
    print(model.config)


if __name__ == '__main__':
    url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original"  # can also be a local file
    model = AutoencoderKL.from_pretrained(url, use_safetensors=True)
    print_model_config(model)