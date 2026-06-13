import pandas as pd
import numpy as np
import umap

import matplotlib.pyplot as plt
from PIL import Image
import os

from sklearn.neighbors import NearestNeighbors

TYPE_MAP = {
    "H": "Hopewell",
    "Hardin": "Hardin",
    "Osceola": "Osceola",
    "O": "Osceola",
    "M": "Merkle",
    "S": "Snyder",
    "Pointphotos": "Unknown",
}

df = pd.read_csv("embeddings/metadata.csv")

cls_embeddings = np.load("embeddings/cls_embedding.npy")
mean_embeddings = np.load("embeddings/mean_embedding.npy")

#cls_embeddings = np.vstack(cls_embeddings)

print(cls_embeddings.shape)

nn = NearestNeighbors(
    n_neighbors=10,
    metric="cosine"
)

nn.fit(mean_embeddings)

query_idx = 2

def extract_type(filename):

    stem = (
        os.path.splitext(
            os.path.basename(filename)
        )[0]
    )

    token = stem.split("_")[-1]

    return TYPE_MAP.get(
        token,
        "Unknown"
    )

df["type"] = df["filename"].apply(
    extract_type
)

print(df["type"].value_counts())

def show_neighbors(indices):
    fig, axes = plt.subplots(
        1,
        len(indices),
        figsize=(20, 4)
    )

    for ax, idx in zip(axes, indices):

        img_path = df.iloc[idx]["filename"]

        img = Image.open(img_path)

        ax.imshow(img)
        ax.axis("off")

        ax.set_title(str(idx))

    plt.tight_layout()
    plt.show()


def inspect_query(query_idx):

    distances, indices = nn.kneighbors(
        mean_embeddings[query_idx].reshape(1, -1)
    )

    print("query:")
    print(df.iloc[query_idx]["filename"])

    print("\nnearest neighbors:")

    for rank, (idx, dist) in enumerate(
        zip(indices[0], distances[0])
    ):
        print(
            rank,
            idx,
            round(dist, 4),
            df.iloc[idx]["filename"]
        )

    show_neighbors(indices[0])

#inspect_query(query_idx)

reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

embedding_2d = reducer.fit_transform(
    mean_embeddings
)

print(embedding_2d.shape)

plt.figure(figsize=(12,12))

plt.scatter(
    embedding_2d[:,0],
    embedding_2d[:,1],
    color="lightgray",
    s=10,
    alpha=0.6,
    zorder=1
)

hardin_mask = (
    df["type"] == "Hardin"
)

plt.scatter(
    embedding_2d[hardin_mask,0],
    embedding_2d[hardin_mask,1],
    s=15
)

snyder_mask = (
    df["type"] == "Snyder"
)

plt.scatter(
    embedding_2d[snyder_mask,0],
    embedding_2d[snyder_mask,1],
    color="red",
    s=15
)

osceola_mask = (
    df["type"] == "Osceola"
)

plt.scatter(
    embedding_2d[osceola_mask,0],
    embedding_2d[osceola_mask,1],
    color="orange",
    s=15
)

merkle_mask = (
    df["type"] == "Merkle"
)

plt.scatter(
    embedding_2d[merkle_mask,0],
    embedding_2d[merkle_mask,1],
    color="green",
    s=15
)

hopewell_mask = (
    df["type"] == "Hopewell"
)

plt.scatter(
    embedding_2d[hopewell_mask,0],
    embedding_2d[hopewell_mask,1],
    color="purple",
    s=15
)


plt.show()