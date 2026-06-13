from transformers import AutoImageProcessor
from transformers import AutoModel

import torch

from PIL import Image
import glob, os

import numpy as np
import pandas as pd

import csv

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

model = AutoModel.from_pretrained("facebook/dinov2-base")

device = "cuda" if torch.cuda.is_available() else "cpu"

model.eval()

model.to(device)

filenames = []
images = []

for infile in glob.glob("pp_scrape/processed_images/*.png"):
    filenames.append(infile)
    images.append(
        Image.open(infile).convert("RGB")
    )

inputs = processor(
    images=images,
    return_tensors="pt"
)

inputs = {
    k: v.to(device) for k, v in inputs.items()
}

with torch.no_grad():
    outputs = model(**inputs)

# print(outputs.last_hidden_state.shape)

# cls_embedding = outputs.last_hidden_state[:,0]

# print(cls_embedding.shape)

# print(cls_embedding[0,:10])  

cls_embedding = (
    outputs.last_hidden_state[:, 0]
    .cpu()
    .numpy()
    .squeeze()
)

mean_embedding = (
    outputs.last_hidden_state[:, 1:]
    .mean(dim=1)
    .cpu()
    .numpy()
    .squeeze()
)

cls_embedding = cls_embedding / np.linalg.norm(
    cls_embedding,
    axis=1,
    keepdims=True
)

mean_embedding = mean_embedding / np.linalg.norm(
    mean_embedding,
    axis=1,
    keepdims=True
)

metadata = pd.DataFrame({
    "filename": filenames
})

metadata.to_csv(
    "embeddings/metadata.csv",
    index=False
)

np.save("embeddings/cls_embedding.npy", cls_embedding)
np.save("embeddings/mean_embedding.npy", mean_embedding)