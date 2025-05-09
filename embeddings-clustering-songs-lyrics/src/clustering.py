from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np


def normalize_embeddings(embeddings):
    return StandardScaler().fit_transform(embeddings)

def perform_dbscan(embedding_vectors, eps=0.5, min_samples=2):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return dbscan.fit_predict(embedding_vectors)

def perform_kmeans(embedding_vectors, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(embedding_vectors)