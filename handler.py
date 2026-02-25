import base64
import gc
import os
import torch
from io import BytesIO
from PIL import Image
from optimum.quanto import QuantizedDiffusersModel
from diffusers import Flux2KleinPipeline
import runpod

device = "cuda"
dtype  = torch.bfloat16

# 1. Network Volume Paths (Serverless uses /runpod-volume)
NETWORK_STORAGE_PATH = "/runpod-volume/models"
BASE_MODEL_PATH = f"{NETWORK_STORAGE_PATH}/FLUX.2-klein-base-9B"
TRANSFORMER_PATH = f"{NETWORK_STORAGE_PATH}/flux2klein-transformer-qint8"

# ── Load once at startup ───────────────────────────────────────────────────
print("Loading pipeline from network storage...")
temp_pipe = Flux2KleinPipeline.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=dtype,
    local_files_only=True # Forces it to use the local files
)

TransformerClass = type(temp_pipe.transformer)
del temp_pipe.transformer
gc.collect()
torch.cuda.empty_cache()

class QuantizedFlux2KleinTransformer(QuantizedDiffusersModel):
    base_class = TransformerClass

print("Loading quantized transformer from network storage...")
temp_pipe.transformer = QuantizedFlux2KleinTransformer.from_pretrained(
    TRANSFORMER_PATH,
    local_files_only=True # Forces it to use the local files
).to(device)

pipe = temp_pipe.to(device)
print("Pipeline ready!")

RATIOS = {
    "1:1":  (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3":  (1152, 896),
    "3:4":  (896, 1152),
}

MAX_SIZE    = (1024, 1024)

def load_and_resize(image_path):
    img = Image.open(image_path)
    img.thumbnail(MAX_SIZE, Image.LANCZOS)
    return img


def base64_to_pil(b64_string):
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")

def handler(job):
    try:
        input_data    = job["input"]
        generation_type = input_data["generation_type"]
        prompt        = input_data["prompt"]
        aspect_ratio  = input_data.get("aspect_ratio", "1:1")
        num_steps     = input_data.get("num_inference_steps", 50)
        guidance      = input_data.get("guidance_scale", 4.0)
        width, height = RATIOS[aspect_ratio]

        if generation_type == "image":

            image_location = input_data["image_path"]

            if not os.path.exists(image_location):
                return {"error": f"Image not found: {image_location}"}

            if aspect_ratio not in RATIOS:
                return {"error": f"Invalid aspect_ratio '{aspect_ratio}'. Valid: {list(RATIOS.keys())}"}
            
            image1 = load_and_resize(image_location)
        
        else :
            try:
                image_b64 = input_data["image"]
                image1 = base64_to_pil(image_b64)           
            except Exception as e:
                return {"error" : f"Error in parsing the image : {e}"}

        result = pipe(
            prompt=prompt,
            image=image1,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=num_steps,
        ).images[0]

        buffer = BytesIO()
        result.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        return {"output": {"image": img_b64}}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})