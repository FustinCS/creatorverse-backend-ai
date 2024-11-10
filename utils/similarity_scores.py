import requests
import torch
from PIL import Image
from io import BytesIO

def load_image_from_url(url):
    """Download image from URL and convert to PIL Image"""
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def calculate_similarity_scores(
        given_url: str, 
        community_urls: list[str],
        model,
        processor,
        cos,
        device
) -> list[float]:
    """Calculate similarity scores between given image and each of a community's images"""
    similarity_scores = []
    
    # calculates similarity scores between given image and each of a communitie's images
    with torch.no_grad():
        given_image = load_image_from_url(given_url)
        inputs = processor(images=given_image, return_tensors="pt")
        given_image_features = model.get_image_features(inputs['pixel_values'].to(device))

        for img_url in community_urls:
            image = load_image_from_url(img_url)
            inputs = processor(images=image, return_tensors="pt")
            image_features = model.get_image_features(inputs['pixel_values'].to(device))
            similarity = cos(given_image_features[0], image_features[0]).item()
            similarity = (similarity + 1) / 2
            similarity_scores.append(similarity)
  
    return similarity_scores