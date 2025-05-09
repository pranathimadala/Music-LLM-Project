import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

from embeddings import get_embedding, load_files
from clustering import normalize_embeddings, perform_kmeans

os.makedirs("outputs", exist_ok=True)

data = load_files("data")

print("Starting embedding and clustering...")

embeddings = {file: get_embedding(text) for file, text in data.items()}
embedding_vectors = np.array(list(embeddings.values()))
embedding_vectors_normalized = normalize_embeddings(embedding_vectors)

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced = tsne.fit_transform(embedding_vectors_normalized)

clusters = perform_kmeans(embedding_vectors_normalized, n_clusters=5)

df = pd.DataFrame({
    'x': reduced[:, 0],
    'y': reduced[:, 1],
    'filename': [f.replace('-', ' ').replace('.txt', '').title() for f in embeddings.keys()],
    'cluster': clusters
})

fig = px.scatter(
    df,
    x="x", y="y",
    color=df["cluster"].astype(str),
    hover_name="filename",
    title="TSNE Lyrics Embedding Clusters",
    labels={"color": "Cluster"}
)

fig.update_traces(
    marker=dict(size=10, opacity=0.8, line=dict(width=1, color="DarkSlateGrey"))
)
fig.update_layout(showlegend=True)
fig.write_html("outputs/cluster_plot.html")
print("Plot saved to outputs/cluster_plot.html")
fig.update_layout(width=1000, height=800)
fig.show()

cluster_results = {lyrics: int(cluster)
                   for lyrics, cluster in zip(embeddings.keys(), clusters)}

output_path = "outputs/clusters.json"
with open(output_path, "w") as f:
    json.dump(cluster_results, f, indent=4)

print("Done. Output saved to outputs/clusters.json")

# Testing:

# for cluster_id in sorted(set(clusters)):
#    print(f"\nCluster {cluster_id}:")
#    for i, label in enumerate(clusters):
#        if label == cluster_id:
#            print("  ", df.iloc[i]['filename'])

# for lyrics, cluster in cluster_results.items():
#     print(f"Lyrics: {lyrics} -> Cluster: {cluster}")