from fastapi import FastAPI, Request
from transformers import CLIPProcessor, CLIPModel
import torch
from utils.similarity_scores import calculate_similarity_scores
import os
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
cos = torch.nn.CosineSimilarity(dim=0)

app = FastAPI()

@app.post("/compare")
async def compareImages(request: Request):
    # fetch the given url from frotnend
    json_body = await request.json()
    url = json_body.get("url")

    # fetch 5 images from each community
    
    # for each community, calculate similarity scores
    similarity_scores = calculate_similarity_scores(url, leaves, model, processor, cos, device)
    
    return {"similarity_scores": similarity_scores}
   
