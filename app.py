import os
import io
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline

# ==============================================================================
# 1. API CONFIGURATION
# ==============================================================================
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
TI_PATH = "models/ghiblivis-TI-PROD/ghiblivisTI.safetensors"
LORA_PATH = "models/ghiblivis-LORA-PROD"
TI_TOKEN = "<ghiblivis-style>"

# Global pipeline variable
pipe = None

# ==============================================================================
# 2. LIFESPAN (Modern FastAPI Startup & Shutdown)
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    global pipe
    print("Booting up inference server and loading base model into VRAM...")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16
        ).to("cuda")
        
        if os.path.exists(TI_PATH):
            pipe.load_textual_inversion(TI_PATH, token=TI_TOKEN)
            print(f"Textual Inversion loaded for token: {TI_TOKEN}")
        else:
            print("TI weights not found. Make sure you trained the TI model!")

        print("Server is ready for inference!")
        
    except Exception as e:
        print(f"Failed to load models: {e}")

    # Hand control over to the FastAPI application to start accepting web requests
    yield 

    # --- SHUTDOWN LOGIC ---
    print("Server shutting down. Clearing VRAM...")
    pipe = None
    torch.cuda.empty_cache()

# Initialize FastAPI with the lifespan context manager
app = FastAPI(
    title="GhibliVis PEFT Inference API",
    description="REST API pitting LoRA against Textual Inversion for style transfer.",
    version="1.0",
    lifespan=lifespan
)

# ==============================================================================
# 3. REQUEST SCHEMAS
# ==============================================================================
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, bad quality, distorted"
    steps: int = 30
    guidance_scale: float = 7.5

# ==============================================================================
# 4. API ENDPOINTS
# ==============================================================================
@app.post("/generate/textual-inversion")
async def generate_ti(request: GenerateRequest):
    """Endpoint for generating images using Textual Inversion."""
    if not pipe:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded yet.")

    final_prompt = request.prompt
    if TI_TOKEN not in final_prompt:
        final_prompt = f"A photo in {TI_TOKEN} style, {request.prompt}"

    print(f"Generating TI Image: {final_prompt}")
    
    image = pipe(
        prompt=final_prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.steps,
        guidance_scale=request.guidance_scale
    ).images[0]

    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    
    return StreamingResponse(memory_stream, media_type="image/png")


@app.post("/generate/lora")
async def generate_lora(request: GenerateRequest):
    """Endpoint for generating images using LoRA."""
    if not pipe:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded yet.")

    print(f"🎨 Generating LoRA Image: {request.prompt}")
    
    try:
        pipe.load_lora_weights(LORA_PATH, weight_name="adapter_model.safetensors")
        
        image = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale
        ).images[0]

    finally:
        # CRITICAL: Always unload LoRA so the base pipeline remains clean
        pipe.unload_lora_weights()

    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    
    return StreamingResponse(memory_stream, media_type="image/png")