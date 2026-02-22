import base64
import gc
import os
import torch
from io import BytesIO
from PIL import Image
from optimum.quanto import QuantizedDiffusersModel
from diffusers import Flux2KleinPipeline
from huggingface_hub import login
from dotenv import load_dotenv
import runpod

load_dotenv()
login(token=os.getenv("HF_TOKEN"))  # ← needed to access gated model

device = "cuda"
dtype  = torch.bfloat16
HF_REPO = "Asjad1020/flux2klein-transformer-qint8"

# ── Load once at startup ───────────────────────────────────────────────────
print("Loading pipeline...")
temp_pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-9B",
    torch_dtype=dtype,
)

TransformerClass = type(temp_pipe.transformer)
del temp_pipe.transformer
gc.collect()
torch.cuda.empty_cache()

class QuantizedFlux2KleinTransformer(QuantizedDiffusersModel):
    base_class = TransformerClass

print("Loading quantized transformer...")
temp_pipe.transformer = QuantizedFlux2KleinTransformer.from_pretrained(HF_REPO).to(device)
pipe = temp_pipe.to(device)
print("Pipeline ready!")

RATIOS = {
    "1:1":  (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3":  (1152, 896),
    "3:4":  (896, 1152),
}

# ── Handler ────────────────────────────────────────────────────────────────
def handler(job):
    try:
        input_data    = job["input"]
        prompt        = input_data["prompt"]
        image_location = input_data["image_path"]
        aspect_ratio  = input_data.get("aspect_ratio", "1:1")
        num_steps     = input_data.get("num_inference_steps", 50)
        guidance      = input_data.get("guidance_scale", 4.0)

        if not os.path.exists(image_location):
            return {"error": f"Image not found: {image_location}"}

        if aspect_ratio not in RATIOS:
            return {"error": f"Invalid aspect_ratio '{aspect_ratio}'. Valid: {list(RATIOS.keys())}"}

        width, height = RATIOS[aspect_ratio]
        image1 = Image.open(image_location)

        result = pipe(
            prompt=prompt,
            image=[image1],
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=num_steps,
        ).images[0]

        buffer = BytesIO()
        result.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        return {"image": img_b64}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})