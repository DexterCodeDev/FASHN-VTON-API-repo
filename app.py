import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from fashn_vton import TryOnPipeline

app = FastAPI(title="Fashn VTON 1.5 API")

# Global pipeline variable
pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    weights_dir = "./weights"
    
    print("Loading Fashn VTON 1.5 pipeline into memory...")
    # The pipeline automatically uses the GPU in bfloat16 precision if available
    pipeline = TryOnPipeline(weights_dir=weights_dir)
    print("Pipeline loaded successfully!")

@app.post("/try-on")
async def try_on(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("tops"), # Options: "tops", "bottoms", "one-pieces"
    garment_photo_type: str = Form("model"), # Options: "model" or "flat-lay"
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is still initializing")
        
    if category not in ["tops", "bottoms", "one-pieces"]:
        raise HTTPException(status_code=400, detail="Invalid category. Must be tops, bottoms, or one-pieces")

    try:
        # Read the uploaded files into memory
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()
        
        # Convert to RGB PIL Images required by the pipeline
        person_pil = Image.open(BytesIO(person_bytes)).convert("RGB")
        garment_pil = Image.open(BytesIO(garment_bytes)).convert("RGB")
        
        # Run inference (takes ~5 seconds on modern GPUs)
        result = pipeline(
            person_image=person_pil,
            garment_image=garment_pil,
            category=category,
            garment_photo_type=garment_photo_type,
            num_timesteps=30,          # 30 is the balanced default
            guidance_scale=1.5,
            segmentation_free=True     # Allows unconstrained garment volume
        )
        
        # Save output image to a buffer and return as an API response
        output_image = result.images[0]
        buf = BytesIO()
        output_image.save(buf, format="PNG")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
