import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from data_pipeline import OptimizedSDDataset
from model_loader import load_base_sd_models

class LoRAConfig:
    model_name = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    output_dir = "ghiblivis-LORA-PROD"
    batch_size = 1
    num_epochs = 10
    learning_rate = 1e-4
    mixed_precision = "fp16"

config = LoRAConfig()
accelerator = Accelerator(mixed_precision=config.mixed_precision)

# 1. Load Models & Add LoRA
tokenizer, text_encoder, vae, unet, noise_scheduler = load_base_sd_models(config.model_name)
lora_config = LoraConfig(r=8, lora_alpha=8, target_modules=["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.1)
unet = get_peft_model(unet, lora_config)

text_encoder.to(accelerator.device)
vae.to(accelerator.device)

# 2. Data Engineering: Load & Cache
train_dataset = OptimizedSDDataset("dataset", tokenizer, metadata_file="train.txt")
train_dataset.cache_latents(vae, accelerator) # Cache to RAM
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# 3. Optimizers
optimizer = torch.optim.AdamW(unet.parameters(), lr=config.learning_rate)
lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * config.num_epochs)

unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

# 4. Training Loop
print("Starting LoRA Training...")
for epoch in range(config.num_epochs):
    unet.train()
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")
    
    for batch in train_dataloader:
        # Grab pre-computed latents directly!
        latents = batch["latent_values"].to(accelerator.device)
        input_ids = batch["input_ids"].to(accelerator.device)

        with accelerator.accumulate(unet):
            with torch.no_grad():
                text_embeddings = text_encoder(input_ids)[0]

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
            loss = F.mse_loss(noise_pred, noise)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())
    progress_bar.close()

# 5. Save Model
accelerator.wait_for_everyone()
unwrapped_unet = accelerator.unwrap_model(unet)
unwrapped_unet.save_pretrained(config.output_dir)
print(f"LoRA weights saved to {config.output_dir}")