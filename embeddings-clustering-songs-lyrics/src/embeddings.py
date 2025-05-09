import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

model = SentenceTransformer("all-mpnet-base-v2")

def get_embedding(text):
    return model.encode(text, normalize_embeddings=True).tolist()

def load_files(directory="data"):
    data = {}
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                data[file] = f.read()
    return data


# @Cache(cache_dir="/tmp")
# def get_embedding(text):
#     response = openai.embeddings.create(
#         model="text-embedding-3-large",
#         input=text,
#     )
#     return response.data[0].embedding


# def load_files(directory="data"):
#     data = {}
#     for file in os.listdir(directory):
#         with open(os.path.join(directory, file), 'r') as f:
#             data[file] = f.read()
#     return data