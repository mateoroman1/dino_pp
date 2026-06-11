from transformers import AutoImageProcessor
from transformers import AutoModel

import torch

from PIL import Image

import numpy as np
import pandas as pd

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

model = AutoModel.from_pretrained("facebook/dinov2-base")

device = "cuda" if torch.cuda.is_available() else "cpu"

model.eval()

model.to(device)

image = Image.open("pp_scrape/processed_images/0f2b0df7be_Osceola.png").convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {
    k: v.to(device) for k, v in inputs.items()
}

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.last_hidden_state.shape)

cls_embedding = outputs.last_hidden_state[:,0]

print(cls_embedding.shape)

print(cls_embedding[0,:10])