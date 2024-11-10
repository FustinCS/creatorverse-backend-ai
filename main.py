from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel
import torch
from utils.similarity_scores import calculate_similarity_score
from dotenv import load_dotenv
import os
from supabase import create_client, Client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
cos = torch.nn.CosineSimilarity(dim=0)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/compare")
async def compareImages(request: Request):
    # fetch the given url from frotnend
    json_body = await request.json()
    url = json_body.get("url")

    # fetch 5 images from each community
    # Query to get each community with up to 5 artworks per community
    response = supabase.from_("Community").select('id, name, Art(id, publicUrl)').limit(5).execute()
    data = response.data

    result = []
    for community in data:
        url_list = []
        for art in community["Art"]:
            url_list.append(art["publicUrl"])
        
        similarity_score = calculate_similarity_score(url, url_list, model, processor, cos, device)
        result.append({"community": community["name"], "similarity_score": similarity_score})
        # Sort the result list by similarity_score in descending order
        sorted_result = sorted(result, key=lambda x: x["similarity_score"], reverse=True)

    return {"result": sorted_result}
   
