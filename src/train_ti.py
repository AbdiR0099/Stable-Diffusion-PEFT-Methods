import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from safetensors.torch import save_file
from tqdm.auto import tqdm

from data_pipeline import OptimizedSDDataset
from model_loader import load_base_sd_models

class TIConfig:
    model_name = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    output_dir = "ghiblivis-TI-PROD"
    placeholder_token = "<ghiblivis-style>"
    initializer_token = "style"
    batch_size = 1
    num_epochs = 15
    learning_rate = 5e-6

config = TIConfig()
accelerator = Accelerator(mixed_precision="fp16")

# 1. Load Models & Inject Token
tokenizer, text_encoder, vae, unet, noise_scheduler = load_base_sd_models(config.model_name)

tokenizer.add_tokens(config.placeholder_token)
placeholder_token_id = tokenizer.convert_tokens_to_ids(config.placeholder_token)
initializer_token_id = tokenizer.convert_tokens_to_ids(config.initializer_token)

text_encoder.resize_token_embeddings(len(tokenizer))
token_embeds = text_encoder.get_input_embeddings().weight.data
token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

# Only train the token embeddings
text_encoder.text_model.embeddings.token_embedding.requires_grad_(True)

vae.to(accelerator.device)
unet.to(accelerator.device)

# 2. Data Engineering: Load & Cache
train_dataset = OptimizedSDDataset("dataset", tokenizer, metadata_file="train.txt", ti_placeholder=config.placeholder_token)
train_dataset.cache_latents(vae, accelerator)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# 3. Optimizers
optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=config.learning_rate)
lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * config.num_epochs)

text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(text_encoder, optimizer, train_dataloader, lr_scheduler)

# 4. Training Loop
print("Starting Textual Inversion Training...")
for epoch in range(config.num_epochs):
    text_encoder.train()
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")
    
    for batch in train_dataloader:
        latents = batch["latent_values"].to(accelerator.device)
        input_ids = batch["input_ids"].to(accelerator.device)

        with accelerator.accumulate(text_encoder):
            text_embeddings = text_encoder(input_ids)[0]
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # UNet forward pass (no grad for UNet, but grad flows to text encoder)
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
            loss = F.mse_loss(noise_pred, noise)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Keep all other tokens frozen except our placeholder
            with torch.no_grad():
                unwrapped_te = accelerator.unwrap_model(text_encoder)
                non_placeholder_indices = torch.arange(len(tokenizer)) != placeholder_token_id
                original_embeds = token_embeds[non_placeholder_indices].to(accelerator.device)
                unwrapped_te.get_input_embeddings().weight[non_placeholder_indices] = original_embeds

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())
    progress_bar.close()

# 5. Save Output
accelerator.wait_for_everyone()
os.makedirs(config.output_dir, exist_ok=True)
unwrapped_te = accelerator.unwrap_model(text_encoder)
learned_embedding = unwrapped_te.get_input_embeddings().weight[placeholder_token_id]

embedding_dict = {config.placeholder_token: learned_embedding.detach().cpu()}
save_file(embedding_dict, os.path.join(config.output_dir, "ghiblivisTI.safetensors"))
print(f" TI Embedding saved to {config.output_dir}")