import pandas as pd
import numpy as np
import ast

from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("embeddings/cls_mean_1.csv")

cls_embeddings = np.load("embeddings/cls_embedding.npy")
mean_embeddings = np.load("embeddings/mean_embedding.npy")

cls_embeddings = np.vstack(cls_embeddings)

print(cls_embeddings.shape)