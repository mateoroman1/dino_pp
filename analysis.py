import pandas as pd
import numpy as np
import ast

import matplotlib.pyplot as plt
from PIL import Image

from sklearn.neighbors import NearestNeighbors

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

inspect_query(query_idx)