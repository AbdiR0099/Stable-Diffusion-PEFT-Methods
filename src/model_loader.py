# model_loader.py
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

def load_base_sd_models(model_name="stable-diffusion-v1-5/stable-diffusion-v1-5"):
    """Loads all base components for Stable Diffusion and freezes them."""
    print(f"Loading Base SD Model: {model_name}...")
    
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")

    # Freeze everything by default (Best practice to prevent accidental gradient tracking)
    for model in [vae, text_encoder, unet]:
        model.requires_grad_(False)
        
    return tokenizer, text_encoder, vae, unet, noise_scheduler