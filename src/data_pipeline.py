import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

class OptimizedSDDataset(Dataset):
    """
    A production-ready PyTorch Dataset for Stable Diffusion fine-tuning.
    Handles efficient image transforms and supports Latent Caching.
    """
    def __init__(self, dataset_path, tokenizer, metadata_file="train.txt", resolution=512, ti_placeholder = None):
        self.dataset_path = Path(dataset_path)
        self.image_dir = self.dataset_path / "images"
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.ti_placeholder = ti_placeholder

        # 1. Load Metadata efficiently
        metadata_path = self.dataset_path / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = [line.strip().split("|") for line in f.readlines() if line.strip()]

        # 2. Native PyTorch Transforms (Lightning fast compared to manual lists)
        self.image_transforms = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(), # Automatically converts HWC -> CHW and normalizes to [0, 1]
            transforms.Normalize([0.5], [0.5]) # Shifts from [0, 1] to [-1, 1] for SD
        ])
        
        # Storage for pre-computed latents
        self.cached_latents = None
        self.cached_text_ids = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        If latents are cached, return them instantly.
        Otherwise, load and transform the image on the fly.
        """
        if self.cached_latents is not None:
            return {
                "latent_values": self.cached_latents[idx],
                "input_ids": self.cached_text_ids[idx]
            }

        # Fallback to on-the-fly processing (Standard ETL)
        img_name, caption = self.metadata[idx]
        image = Image.open(self.image_dir / img_name).convert("RGB")
        
        # Apply C++ optimized transforms
        pixel_values = self.image_transforms(image)
        
        if self.ti_placeholder:
            final_caption = f"A photo in {self.ti_placeholder} style showing {caption}"
        else:
            final_caption = caption
        # Tokenize text
        text_inputs = self.tokenizer(
            final_caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids.squeeze()
        }

    @torch.no_grad()
    def cache_latents(self, vae, accelerator):
        """
        Pre-computes all image latents through the VAE before training starts.
        This saves massive amounts of VRAM and drastically speeds up the training loop.
        """
        accelerator.print("Pre-computing latents to optimize training loop...")
        vae.to(accelerator.device)
        vae.eval()
        
        latents_list = []
        text_ids_list = []
        
        for idx in tqdm(range(len(self)), disable=not accelerator.is_local_main_process):
            # Load raw transformed data
            data = self.__getitem__(idx)
            
            # Move to GPU
            pixel_values = data["pixel_values"].unsqueeze(0).to(accelerator.device, dtype=vae.dtype)
            
            # Encode image to latent space
            latent_dist = vae.encode(pixel_values).latent_dist
            latent = latent_dist.sample() * vae.config.scaling_factor
            
            # Store in CPU RAM (to save GPU memory)
            latents_list.append(latent.squeeze(0).cpu())
            text_ids_list.append(data["input_ids"].cpu())
            
        self.cached_latents = latents_list
        self.cached_text_ids = text_ids_list
        accelerator.print("Latent caching complete!")